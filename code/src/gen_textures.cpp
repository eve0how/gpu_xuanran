// 文件说明：离线生成展示用程序化纹理到 textures/ 目录。
// 原创性声明：参考已有代码（Texture 生成接口），命令行入口独立实现。

#include "texture.hpp"
#include <cstdio>

static int runShowcaseGeneration(const char *outDir) {
    if (!Texture::generateShowcaseTextures(outDir)) {
        fprintf(stderr, "Showcase texture generation failed.\n");
        return 1;
    }
    printf("Wrote showcase textures to %s/\n", outDir);
    return 0;
}

int main() {
    return runShowcaseGeneration("textures");
}
