# DirectX 12 Sandbox
This repository is for storing projects/demos I've created while learning DirectX 12 by working through _Introduction to 3D Game Programming With DirectX 12_ by Frank D. Luna. Luna provides a set of "framework" code files with the book to abstract boilerplate code, and this repo includes those files in addition to files containing source code that I wrote myself while working through the chapters and exercises. Files included here that were provided by the author are unmodified and all contain their original header attributions to Frank Luna. 

## Examples

_RainbowBoxDemo.cpp:_

![RainbowCube][rainbowcube]

_ShapesDemo.cpp:_

![Shapes][shapes]

_RiverDemo.cpp:_

![River][river]

[rainbowcube]: img/RainbowCube.gif
[shapes]: img/ShapesDemo.gif
[river]: img/RiverDemo.gif


## Notes:

* For VS2019 v16.8 or newer, must use /permissive option at end of command line to offset strict conformance mode (/permissive-): https://developercommunity.visualstudio.com/t/1681-breaks-compilation/1255614?preview=true

* Linker->System->SubSystem should be set to WINDOWS for the linker to choose the correct entry point symbol.

* HLSL files should be excluded from build since we compile them in code with D3DCompileFromFile()

* Seperate demos are stored in the "{Name}Demo.cpp" classes. Each one contains a WinMain method, but only one can exist to compile. To run a particular demo, comment out the WinMain method in all of the other Demo files.

## References:
_Introduction to 3D Game Programming With DirectX 12_ by Frank D. Luna

_Foundations of Game Engine Development Volume 1: Mathematics_ by Eric Lengyel

_Foundations of Game Engine Development Volume 2: Rendering_ by Eric Lengyel