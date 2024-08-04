import Metal

public extension Compute {
    /// A structure that holds and manages arguments for a compute pipeline.
    ///
    /// This structure uses dynamic member lookup to provide a convenient way to access and set arguments.
    @dynamicMemberLookup
    struct Arguments {
        /// The underlying dictionary storing the arguments.
        internal var arguments: [String: Argument]

        /// Provides access to arguments using dynamic member lookup.
        ///
        /// - Parameter name: The name of the argument.
        /// - Returns: The argument value if it exists, or nil if it doesn't.
        public subscript(dynamicMember name: String) -> Argument? {
            get {
                arguments[name]
            }
            set {
                arguments[name] = newValue
                // NOTE: It would be nice to assign name as a label to buffers/textures that have no name.
            }
        }
    }

    /// Represents an argument that can be passed to a compute shader.
    ///
    /// This struct encapsulates the logic for encoding the argument to a compute command encoder
    /// and setting it as a constant value in a function constant values object.
    struct Argument {
        /// A closure that encodes the argument to a compute command encoder.
        internal var encode: (MTLComputeCommandEncoder, Int) -> Void

        /// A closure that sets the argument as a constant value in a function constant values object.
        internal var constantValue: (MTLFunctionConstantValues, String) -> Void

        internal init(encode: @escaping (MTLComputeCommandEncoder, Int) -> Void, constantValue: @escaping (MTLFunctionConstantValues, String) -> Void) {
            self.encode = encode
            self.constantValue = constantValue
        }

        /// Creates an integer argument.
        ///
        /// - Parameter value: The integer value.
        /// - Returns: An `Argument` instance representing the integer.
        public static func int(_ value: some BinaryInteger) -> Self {
            .init { encoder, index in
                withUnsafeBytes(of: value) { buffer in
                    guard let baseAddress = buffer.baseAddress else {
                        fatalError("Could not get baseAddress.")
                    }
                    encoder.setBytes(baseAddress, length: buffer.count, index: index)
                }
            }
            constantValue: { constants, name in
                withUnsafeBytes(of: value) { buffer in
                    guard let baseAddress = buffer.baseAddress else {
                        fatalError("Could not get baseAddress.")
                    }
                    switch value {
                    case is Int8:
                        constants.setConstantValue(baseAddress, type: .char, withName: name)
                    case is UInt8:
                        constants.setConstantValue(baseAddress, type: .uchar, withName: name)
                    case is Int16:
                        constants.setConstantValue(baseAddress, type: .short, withName: name)
                    case is UInt16:
                        constants.setConstantValue(baseAddress, type: .ushort, withName: name)
                    case is Int32:
                        constants.setConstantValue(baseAddress, type: .int, withName: name)
                    case is UInt32:
                        constants.setConstantValue(baseAddress, type: .uint, withName: name)
                    default:
                        fatalError("Unsupported integer type.")
                    }
                }
            }
        }

        /// Creates a float argument.
        ///
        /// - Parameter value: The float value.
        /// - Returns: An `Argument` instance representing the float.
        public static func float(_ value: Float) -> Self {
            .init { encoder, index in
                withUnsafeBytes(of: value) { buffer in
                    guard let baseAddress = buffer.baseAddress else {
                        fatalError("Could not get baseAddress.")
                    }
                    encoder.setBytes(baseAddress, length: buffer.count, index: index)
                }
            }
            constantValue: { constants, name in
                withUnsafeBytes(of: value) { buffer in
                    guard let baseAddress = buffer.baseAddress else {
                        fatalError("Could not get baseAddress.")
                    }
                    constants.setConstantValue(baseAddress, type: .float, withName: name)
                }
            }
        }

        /// Creates a boolean argument.
        ///
        /// - Parameter value: The boolean value.
        /// - Returns: An `Argument` instance representing the boolean.
        public static func bool(_ value: Bool) -> Self {
            .init { encoder, index in
                withUnsafeBytes(of: value) { buffer in
                    guard let baseAddress = buffer.baseAddress else {
                        fatalError("Could not get baseAddress.")
                    }
                    encoder.setBytes(baseAddress, length: buffer.count, index: index)
                }
            }
            constantValue: { constants, name in
                withUnsafeBytes(of: value) { buffer in
                    guard let baseAddress = buffer.baseAddress else {
                        fatalError("Could not get baseAddress.")
                    }
                    constants.setConstantValue(baseAddress, type: .bool, withName: name)
                }
            }
        }

        /// Creates a buffer argument.
        ///
        /// - Parameters:
        ///   - buffer: The Metal buffer to be used as an argument.
        ///   - offset: The offset within the buffer. Defaults to 0.
        /// - Returns: An `Argument` instance representing the buffer.
        public static func buffer(_ buffer: MTLBuffer, offset: Int = 0) -> Self {
            .init { encoder, index in
                encoder.setBuffer(buffer, offset: offset, index: index)
            }
            constantValue: { _, _ in
                fatalError("Unimplemented")
            }
        }

        /// Creates a texture argument.
        ///
        /// - Parameter texture: The Metal texture to be used as an argument.
        /// - Returns: An `Argument` instance representing the texture.
        public static func texture(_ texture: MTLTexture) -> Self {
            Self { encoder, index in
                encoder.setTexture(texture, index: index)
            }
            constantValue: { _, _ in
                fatalError("Unimplemented")
            }
        }
    }
}
