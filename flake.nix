{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs";

  nixConfig = {
    substituters = [
     "https://cache.nixos-cuda.org"
    ];
    extra-trusted-public-keys = [
      "cache.nixos-cuda.org:74DUi4Ye579gUqzH4ziL9IyiJBlDpMRn9MBN8oNan9M="
    ];
  };

  outputs = { nixpkgs, ... }: let
    inherit (nixpkgs) lib;

    pkgs.config = {
      allowUnfree = true;
      cudaSupport = true;
    };

    get-pkgs = system: import nixpkgs {
      inherit system;
      inherit (pkgs) config;
    };

    forAllSystems = function:
      lib.genAttrs
        [ "x86_64-linux" ]
        (system: function (get-pkgs system));
    in
    {
      devShells = forAllSystems (pkgs: let
        pyenv = pkgs.python314.withPackages (ps: with ps; [
          numpy
          triton-cuda
          cuda-tile
          cupy
          pytest
          torch
          ruff
        ]);
      in {
        default = pkgs.mkShell {
          packages = [
            pyenv
            pkgs.cudaPackages.cuda_tileiras
            pkgs.cudaPackages.cuda_nvcc
          ];
        };
      });
    };
}
