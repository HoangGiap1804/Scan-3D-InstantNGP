{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

let
  pythonPackages = pkgs.python3Packages;
  cudaToolkit = pkgs.cudaPackages.cudatoolkit;
  cudnn = pkgs.cudaPackages.cudnn;
in
pkgs.mkShell {
  name = "torch-ngp-env";

  buildInputs = with pkgs; [
    # CUDA
    cudaToolkit
    cudnn
    cudaPackages.cuda_nvcc
    cudaPackages.cuda_cudart
    cudaPackages.cuda_cccl
    cudaPackages.cuda_nvrtc
    
    # Python
    python310
    python310Packages.pip
    python310Packages.virtualenv
    
    # Build tools
    cmake
    ninja
    pkg-config
    gcc
    git
    
    # Video & Image processing
    ffmpeg
    colmap
    
    # Libraries for OpenCV and DearPyGui
    libGL
    libGLU
    xorg.libX11
    xorg.libXcursor
    xorg.libXext
    xorg.libXinerama
    xorg.libXi
    xorg.libXrandr
    xorg.libxcb
    xorg.libXrender
    libsm
    libice
    glib
    zlib
    stdenv.cc.cc.lib
  ];

  shellHook = ''
    export CUDA_PATH=${cudaToolkit}
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.libGL
      pkgs.glib
      pkgs.zlib
      cudaToolkit
      cudnn
      pkgs.xorg.libX11
      pkgs.xorg.libXcursor
      pkgs.xorg.libXext
      pkgs.xorg.libXinerama
      pkgs.xorg.libXi
      pkgs.xorg.libXrandr
      pkgs.xorg.libxcb
      pkgs.xorg.libXrender
      pkgs.libsm
      pkgs.libice
    ]}:$LD_LIBRARY_PATH
    
    # Setup virtualenv
    VENV=.venv
    if [ ! -d "$VENV" ]; then
        python -m venv "$VENV"
    fi
    source "$VENV/bin/activate"
    
    # Set CUDA_HOME for torch extensions build
    export CUDA_HOME=${pkgs.cudaPackages.cuda_nvcc}
    export CUDACXX=${pkgs.cudaPackages.cuda_nvcc}/bin/nvcc

    export CC=$(which gcc)
    export CXX=$(which g++)
    export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH
    
    echo "Nix environment for torch-ngp loaded!"
    echo "Run 'pip install -r requirements.txt' to install Python dependencies."
    echo "To build tiny-cuda-nn, ensure you have the git submodule initialized."
  '';
}
