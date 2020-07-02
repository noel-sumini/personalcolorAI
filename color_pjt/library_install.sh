wget https://github.com/Itseez/opencv/archive/2.4.11.zip
unzip 2.4.11.zip
cd opencv-2.4.11
mkdir release
cd release
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
make
make install


wget https://github.com/davisking/dlib/releases/download/v18.16/dlib-18.16.tar.bz2
bzip2 -d dlib-18.16.tar.bz2
tar -xvf dlib-18.16.tar
cd dlib-18.16/
cd python_examples/
mkdir build
cd build
cmake ../../tools/python
cmake --build . --config Release
cp dlib.so /usr/local/lib/python2.7/dist-packages/