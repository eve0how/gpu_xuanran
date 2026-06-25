#include "texture.hpp"
#include <cstdio>

int main() {
    if (!Texture::generateShowcaseTextures("textures")) {
        fprintf(stderr, "Failed to generate showcase textures.\n");
        return 1;
    }
    printf("Generated textures in textures/\n");
    return 0;
}
