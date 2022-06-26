import ctypes

rust_lib_name = "C:\\Users\\Administrator\\Desktop\\práce\\DroneProject\\code_examples\\rust_extension_for_py_example\\target\\release\\rust_lib.dll"
rust = ctypes.CDLL(rust_lib_name)


if __name__ == "__main__":
    SOME_BYTES = "Python says hi inside Rust!".encode("utf-8")
    rust.print_string(SOME_BYTES)
