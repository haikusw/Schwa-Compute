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
            texture2d<float, access::write> texture [[texture(0)]],
            constant uint &max_iterations [[buffer(3)]],
            constant float &x_min [[buffer(4)]],
            constant float &x_max [[buffer(5)]],
            constant float &y_min [[buffer(6)]],
            constant float &y_max [[buffer(7)]]
    ) {
        const uint j = gid.x;
        const uint i = gid.y;
        auto c = Complex(
            x_min + (float(j) / float(texture.get_width() - 1)) * (x_max - x_min),
            y_min + (float(i) / float(texture.get_height() - 1)) * (y_max - y_min)
        );

        auto z = Complex();
        uint iteration = 0;
        while (z.magnitude_squared() <= 4.0 && iteration < max_iterations) {
            z = z.mul_add(c);
            iteration += 1;
        }
        // Normalize the iteration count to [0, 1]
        auto value = float(iteration) / float(max_iterations);
        texture.write(value, gid);

        }
    """#

    static func main() async throws {
        let device = MTLCreateSystemDefaultDevice()!
        let logger = Logger()

        let width: UInt32 = 4096
        let height: UInt32 = 4096

        // Mandelbrot parameters
        let max_iterations: UInt32 = 1024
        let x_min: Float = -2.0
        let x_max: Float = 1.0
        let y_min: Float = -1.5
        let y_max: Float = 1.5

        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.usage = .shaderWrite
        textureDescriptor.pixelFormat = .r32Float
        textureDescriptor.width = Int(width)
        textureDescriptor.height = Int(height)

        let texture = device.makeTexture(descriptor: textureDescriptor)!

        let compute = try Compute(device: device, logger: logger)
        let library = ShaderLibrary.source(source, enableLogging: true)
        var pipeline = try compute.makePipeline(function: library.hello_world)
        pipeline.arguments.texture = .texture(texture)
        pipeline.arguments.max_iterations = .int(max_iterations)
        pipeline.arguments.x_min = .float(x_min)
        pipeline.arguments.x_max = .float(x_max)
        pipeline.arguments.y_min = .float(y_min)
        pipeline.arguments.y_max = .float(y_max)

        let w = Int(sqrt(Double(pipeline.maxTotalThreadsPerThreadgroup)))
        try timeit {
            try compute.run(pipeline: pipeline, threads: .init(Int(width), Int(height), 1), threadsPerThreadgroup: .init(width: w, height: w, depth: 1))
        }

        try texture.export(to: URL(filePath: "/tmp/mandelbrot.png"), reveal: true)
    }
}
