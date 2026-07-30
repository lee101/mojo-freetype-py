"""Anti-aliased scan conversion for flattened glyph outlines."""

from std.algorithm import parallelize
from std.gpu.host import DeviceContext
from std.math import floor
from std.sys.info import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]


def add_span(
    coverage: FPtr,
    x_start: Float64,
    x_end: Float64,
    left: Int,
    width: Int,
):
    var lo = max(x_start, Float64(left))
    var hi = min(x_end, Float64(left + width))
    if hi <= lo:
        return
    var first = Int(floor(lo)) - left
    var last = Int(floor(hi - 1.0e-12)) - left
    if first < 0:
        first = 0
    if last >= width:
        last = width - 1
    var first_lo = Float64(left + first)
    var first_overlap = min(hi, first_lo + 1.0) - max(lo, first_lo)
    if first_overlap > 0.0:
        coverage[first] += first_overlap
    if first == last:
        return

    comptime W = simd_width_of[DType.float64]()
    var col = first + 1
    while col + W <= last:
        var values = coverage.load[width=W](col)
        coverage.store(col, values + 1.0)
        col += W
    while col < last:
        coverage[col] += 1.0
        col += 1

    var last_lo = Float64(left + last)
    var last_overlap = min(hi, last_lo + 1.0) - max(lo, last_lo)
    if last_overlap > 0.0:
        coverage[last] += last_overlap


def raster_gray_row(
    segments: FPtr,
    edge_count: Int,
    intersections: FPtr,
    directions: IPtr,
    active_edges: IPtr,
    coverage: FPtr,
    bitmap: BPtr,
    width: Int,
    height: Int,
    pitch: Int,
    left: Int,
    top: Int,
    samples: Int,
    even_odd: Bool,
    row: Int,
    scratch_row: Int,
):
    var row_intersections = intersections + scratch_row * edge_count
    var row_directions = directions + scratch_row * edge_count
    var row_active_edges = active_edges + scratch_row * edge_count
    var row_coverage = coverage + scratch_row * width

    comptime W = simd_width_of[DType.float64]()
    var col = 0
    while col + W <= width:
        row_coverage.store(col, SIMD[DType.float64, W](0.0))
        col += W
    while col < width:
        row_coverage[col] = 0.0
        col += 1

    var row_top = Float64(top - row)
    var row_bottom = row_top - 1.0
    var candidate_count = 0
    var edge = 0
    while edge + W <= edge_count:
        var y_min = segments.load[width=W](edge_count + edge)
        var y_max = segments.load[width=W](2 * edge_count + edge)
        var top_values = SIMD[DType.float64, W](row_top)
        var bottom_values = SIMD[DType.float64, W](row_bottom)
        var active = y_min.lt(top_values) & y_max.gt(bottom_values)
        comptime for lane in range(W):
            if active[lane]:
                row_active_edges[candidate_count] = Int32(edge + lane)
                candidate_count += 1
        edge += W
    while edge < edge_count:
        if (
            segments[edge_count + edge] < row_top
            and segments[2 * edge_count + edge] > row_bottom
        ):
            row_active_edges[candidate_count] = Int32(edge)
            candidate_count += 1
        edge += 1

    var inverse_samples = 1.0 / Float64(samples)
    for sample in range(samples):
        var y = Float64(top - row) - (Float64(sample) + 0.5) * inverse_samples
        var count = 0
        for candidate in range(candidate_count):
            edge = Int(row_active_edges[candidate])
            var y_min = segments[edge_count + edge]
            var y_max = segments[2 * edge_count + edge]
            if y_min <= y and y < y_max:
                row_intersections[count] = (
                    segments[edge]
                    + (y - y_min) * segments[3 * edge_count + edge]
                )
                row_directions[count] = Int32(segments[4 * edge_count + edge])
                count += 1

        for i in range(1, count):
            var x = row_intersections[i]
            var direction = row_directions[i]
            var j = i
            while j > 0 and row_intersections[j - 1] > x:
                row_intersections[j] = row_intersections[j - 1]
                row_directions[j] = row_directions[j - 1]
                j -= 1
            row_intersections[j] = x
            row_directions[j] = direction

        var winding = 0
        var span_start = 0.0
        for i in range(count):
            var was_inside = (winding & 1) != 0 if even_odd else winding != 0
            winding += 1 if even_odd else Int(row_directions[i])
            var is_inside = (winding & 1) != 0 if even_odd else winding != 0
            if not was_inside and is_inside:
                span_start = row_intersections[i]
            elif was_inside and not is_inside:
                add_span(
                    row_coverage,
                    span_start,
                    row_intersections[i],
                    left,
                    width,
                )

    var scale = 255.0 * inverse_samples
    col = 0
    while col + W <= width:
        var values = row_coverage.load[width=W](col) * scale
        values = min(
            values,
            SIMD[DType.float64, W](255.0),
        )
        values = max(
            values,
            SIMD[DType.float64, W](0.0),
        )
        comptime for lane in range(W):
            bitmap[row * pitch + col + lane] = UInt8(
                Int(floor(values[lane] + 0.5))
            )
        col += W
    while col < width:
        var value = row_coverage[col] * scale
        value = min(255.0, max(0.0, value))
        bitmap[row * pitch + col] = UInt8(Int(floor(value + 0.5)))
        col += 1


def raster_gray(
    segments: FPtr,
    edge_count: Int,
    intersections: FPtr,
    directions: IPtr,
    active_edges: IPtr,
    coverage: FPtr,
    bitmap: BPtr,
    width: Int,
    height: Int,
    pitch: Int,
    left: Int,
    top: Int,
    samples: Int,
    even_odd: Bool,
    use_parallel: Bool,
):
    if width <= 0 or height <= 0 or samples <= 0:
        return

    if use_parallel:
        try:
            var context = DeviceContext(api="cpu")
            context.set_as_current()

            @parameter
            def render_row(row: Int):
                raster_gray_row(
                    segments,
                    edge_count,
                    intersections,
                    directions,
                    active_edges,
                    coverage,
                    bitmap,
                    width,
                    height,
                    pitch,
                    left,
                    top,
                    samples,
                    even_odd,
                    row,
                    row,
                )

            parallelize[render_row](height)
        except:
            for row in range(height):
                raster_gray_row(
                    segments,
                    edge_count,
                    intersections,
                    directions,
                    active_edges,
                    coverage,
                    bitmap,
                    width,
                    height,
                    pitch,
                    left,
                    top,
                    samples,
                    even_odd,
                    row,
                    row,
                )
    else:
        for row in range(height):
            raster_gray_row(
                segments,
                edge_count,
                intersections,
                directions,
                active_edges,
                coverage,
                bitmap,
                width,
                height,
                pitch,
                left,
                top,
                samples,
                even_odd,
                row,
                0,
            )


def pack_mono(
    gray: BPtr,
    mono: BPtr,
    width: Int,
    height: Int,
    gray_pitch: Int,
    mono_pitch: Int,
):
    comptime W = 8
    for row in range(height):
        var row_gray = gray + row * gray_pitch
        var row_mono = mono + row * mono_pitch
        var col = 0
        var byte = 0
        while col + W <= width:
            var values = row_gray.load[width=W](col)
            var set_bits = values.ge(SIMD[DType.uint8, W](128))
            var packed = UInt8(0)
            comptime for lane in range(W):
                if set_bits[lane]:
                    packed = packed | UInt8(1 << (7 - lane))
            row_mono[byte] = packed
            col += W
            byte += 1
        if col < width:
            var packed = UInt8(0)
            while col < width:
                if row_gray[col] >= 128:
                    packed = packed | UInt8(1 << (7 - col % 8))
                col += 1
            row_mono[byte] = packed
            byte += 1
        while byte < mono_pitch:
            row_mono[byte] = 0
            byte += 1


@export("mft_raster_gray")
def mft_raster_gray(
    segments_addr: Int,
    edge_count: Int,
    intersections_addr: Int,
    directions_addr: Int,
    active_edges_addr: Int,
    coverage_addr: Int,
    bitmap_addr: Int,
    width: Int,
    height: Int,
    pitch: Int,
    left: Int,
    top: Int,
    samples: Int,
    even_odd: Int,
    use_parallel: Int,
) abi("C"):
    raster_gray(
        FPtr(unsafe_from_address=segments_addr),
        edge_count,
        FPtr(unsafe_from_address=intersections_addr),
        IPtr(unsafe_from_address=directions_addr),
        IPtr(unsafe_from_address=active_edges_addr),
        FPtr(unsafe_from_address=coverage_addr),
        BPtr(unsafe_from_address=bitmap_addr),
        width,
        height,
        pitch,
        left,
        top,
        samples,
        even_odd != 0,
        use_parallel != 0,
    )


@export("mft_pack_mono")
def mft_pack_mono(
    gray_addr: Int,
    mono_addr: Int,
    width: Int,
    height: Int,
    gray_pitch: Int,
    mono_pitch: Int,
) abi("C"):
    pack_mono(
        BPtr(unsafe_from_address=gray_addr),
        BPtr(unsafe_from_address=mono_addr),
        width,
        height,
        gray_pitch,
        mono_pitch,
    )


@export("mft_simd_width_float64")
def mft_simd_width_float64() abi("C") -> Int:
    return simd_width_of[DType.float64]()
