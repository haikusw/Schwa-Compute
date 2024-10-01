import Compute
import Metal
import os

enum MandelbrotDemo: Demo {
    static let source = #"""
        #include <metal_stdlib>
        #include <metal_logging>

        using namespace metal;

        struct Complex {
            float real;
            float imag;

            // Default constructor
            Complex() : real(0.0), imag(0.0) {}

            // Parameterized constructor
            Complex(float r, float i) : real(r), imag(i) {}

            // Method to compute z = self * self + other
            Complex mul_add(const Complex other) const {
                // (a + bi)^2 + (c + di) = (a^2 - b^2 + c) + (2ab + d)i
                            float r = real * real - imag * imag + other.real;
                            float i = 2.0 * real * imag + other.imag;
                return Complex(r, i);
            }

            // Method to compute the squared magnitude
            float magnitude_squared() const {
                return real * real + imag * imag;
            }
        };

        uint2 gid [[thread_position_in_grid]];


        kernel void hello_world(
            device float *buffer [[buffer(0)]],
            constant uint &width [[buffer(1)]],
            constant uint &height [[buffer(2)]],
            constant uint &max_iterations [[buffer(3)]],
            constant float &x_min [[buffer(4)]],
            constant float &x_max [[buffer(5)]],
            constant float &y_min [[buffer(6)]],
            constant float &y_max [[buffer(7)]]
    ) {
        const uint j = gid.x;
        const uint i = gid.y;
        auto c = Complex(
            x_min + (float(j) / float(width - 1)) * (x_max - x_min),
            y_min + (float(i) / float(height - 1)) * (y_max - y_min)
        );

        auto z = Complex();
        uint iteration = 0;
        while (z.magnitude_squared() <= 4.0 && iteration < max_iterations) {
            z = z.mul_add(c);
            iteration += 1;
        }
        // Normalize the iteration count to [0, 1]
        buffer[i + j * width] = float(iteration) / float(max_iterations);
        }
    """#

    static func main() async throws {
        let device = MTLCreateSystemDefaultDevice()!
        let logger = Logger()

        let width: UInt32 = 800;
        let height: UInt32 = 800;

        // Mandelbrot parameters
        let max_iterations: UInt32 = 1000
        let x_min: Float = -2.0
        let x_max: Float = 1.0
        let y_min: Float = -1.5
        let y_max: Float = 1.5

        let buffer = device.makeBuffer(length: MemoryLayout<Float>.size * Int(width) * Int(height), options: [])!
        let compute = try Compute(device: device, logger: logger)
        let library = ShaderLibrary.source(source, enableLogging: true)
        var pipeline = try compute.makePipeline(function: library.hello_world)
        pipeline.arguments.buffer = .buffer(buffer)
        pipeline.arguments.width = .int(width)
        pipeline.arguments.height = .int(height)
        pipeline.arguments.max_iterations = .int(max_iterations)
        pipeline.arguments.x_min = .float(x_min)
        pipeline.arguments.x_max = .float(x_max)
        pipeline.arguments.y_min = .float(y_min)
        pipeline.arguments.y_max = .float(y_max)

        let w = Int(sqrt(Double(pipeline.maxTotalThreadsPerThreadgroup)))
        try compute.run(pipeline: pipeline, threads: .init(Int(width), Int(height), 1), threadsPerThreadgroup: .init(width: w, height: w, depth: 1))
    }
}
