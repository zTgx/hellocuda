#include "cxx_17.h"

int main() {
    MyStruct ms;

    auto [u, v] = ms;

    std::cout << "u = " << u << ", v = " << v << std::endl;

    auto [i, s] = getStruct();
    std::cout << "i = " << i << ", s = " << s << std::endl;

    std::cout << "trace: " << std::boolalpha << MyStruct::trace << std::endl;


    [[maybe_unused]]int a = 1;

    switch (a) {
        case 1:
            std::cout << "a" << std::endl;
            [[fallthrough]];
        case 2:
            std::cout << ", 2" << std::endl;
            break;

        default:
            std::cout << "default" << std::endl;
            break;
    }    

    std::cout << "is a: " << std::boolalpha << A::B::isa << std::endl;


    return 0;
}
