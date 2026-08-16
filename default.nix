{
  buildPythonPackage,
  hatchling,
  gitignoreSource,
  lib,
  numpy,
  pytestCheckHook,
}:
let
  versionFile = builtins.readFile ./src/ame_py/__init__.py;
  versionLine = builtins.replaceStrings [ "\n" "\r" ] [ "" "" ] versionFile;
  versionMatch = builtins.match ''.*__version__ = "([^"]+)".*'' versionLine;
  version = builtins.head versionMatch;
in
buildPythonPackage {
  pname = "ame-py";
  inherit version;
  src = gitignoreSource ./.;
  pyproject = true;

  build-system = [ hatchling ];
  nativeBuildInputs = [ ];
  dependencies = [ numpy ];

  doCheck = true;
  nativeCheckInputs = [ pytestCheckHook ];
  enabledTestPaths = [ "tests" ];
  pythonImportsCheck = [ "ame_py" ];

  meta = {
    description = "";
    homepage = "https://github.com/yushengyangchem/ame-py";
    license = lib.licenses.mit;
    maintainers = [
      {
        name = "Yusheng Yang";
        email = "yushengyangchem@gmail.com";
        github = "yushengyangchem";
      }
    ];
  };
}
