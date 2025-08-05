#include <iostream>

struct MyStruct {
    static const bool trace = false;

    int i = 0;
    std::string s{"hello"};
};

MyStruct getStruct() {
    return MyStruct{42, "world."};
}

namespace A {
    namespace B {
        static const bool isb = true;
        const static bool isa = false;

        // in cxx20 -> char8_t
        const static char me = u8'o';
    }
}

