{
  pkgs,
  pythonVersion,
  gitignoreSource,
}:
let
  pythonEnv = pkgs.${pythonVersion}.override {
    packageOverrides = self: super: {
      ame-py = self.callPackage ./. { inherit gitignoreSource; };
    };
  };
in
pythonEnv
