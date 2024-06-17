NS3_VERSION="ns-3.31"

# Usage help
if [ "$1" == "--help" ]; then
  echo "Usage: bash build.sh [--help, --debug_all, --debug_minimal, --optimized, --optimized_with_tests]"
  exit 0
fi

cd simulator || exit 1

# Configure the build
if [ "$1" == "--debug_all" ]; then
CXX=x86_64-linux-gnu-g++-9  ./waf configure --enable-sudo --build-profile=debug --enable-mpi --enable-examples --enable-tests --enable-gcov --out=build/debug_all || exit 1

elif [ "$1" == "--debug_minimal" ]; then
CXX=x86_64-linux-gnu-g++-9  ./waf configure --enable-sudo --build-profile=debug --enable-mpi --out=build/debug_minimal || exit 1

elif [ "$1" == "--optimized" ]; then
CXX=x86_64-linux-gnu-g++-9  ./waf configure --enable-sudo --build-profile=optimized  --enable-examples --enable-mpi --out=build/optimized || exit 1

elif [ "$1" == "--optimized_with_tests" ]; then
CXX=x86_64-linux-gnu-g++-9  ./waf configure --enable-sudo --build-profile=optimized --enable-mpi --enable-tests --out=build/optimized_with_tests || exit 1

elif [ "$1" == "" ]; then
  # Default is debug_all
CXX=x86_64-linux-gnu-g++-9  ./waf configure --enable-sudo --build-profile=debug --enable-mpi --enable-examples --enable-tests --enable-gcov --out=build/debug_all || exit 1

else
  echo "Invalid build option: $1"
  echo "Usage: bash build.sh [--debug_all, --debug_minimal, --optimized, --optimized_with_tests]"
  exit 1
fi

# Perform the build
CXX=x86_64-linux-gnu-g++-9 ./waf build || exit 1
