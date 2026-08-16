{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
    git-hooks = {
      url = "github:cachix/git-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    gitignore = {
      url = "github:hercules-ci/gitignore.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs =
    { flake-parts, gitignore, ... }@inputs:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ inputs.git-hooks.flakeModule ];
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      perSystem =
        {
          config,
          self',
          pkgs,
          lib,
          ...
        }:
        let
          mkPythonEnv =
            pythonVersion:
            (import ./python-env.nix {
              inherit pkgs pythonVersion;
              inherit (gitignore.lib) gitignoreSource;
            });
          pyVersions = [
            "python3"
            "python311"
            "python312"
            "python313"
            "python314"
            "python315"
          ];
        in
        {
          legacyPackages = lib.genAttrs pyVersions (v: mkPythonEnv v);
          packages = {
            default = self'.legacyPackages.python3.pkgs.ame-py;
          }
          // builtins.listToAttrs (
            lib.forEach pyVersions (v: {
              name = "ame-py-${v}";
              value = self'.legacyPackages.${v}.pkgs.ame-py;
            })
          );
          devShells.default = pkgs.mkShell {
            inputsFrom = [ config.pre-commit.devShell ];
            packages = [
              (self'.legacyPackages.python3.withPackages (
                p: with p; [
                  ame-py
                  pytest
                  build
                  twine
                ]
              ))
            ]
            ++ (with pkgs; [
              just
            ]);
          };

          pre-commit = {
            check.enable = true;
            settings.hooks = {
              nixfmt.enable = true;
              ruff-format.enable = true;
              taplo.enable = true;
              prettier = {
                enable = true;
                excludes = [ "flake.lock" ];
              };
            };
          };
        };
    };
}
