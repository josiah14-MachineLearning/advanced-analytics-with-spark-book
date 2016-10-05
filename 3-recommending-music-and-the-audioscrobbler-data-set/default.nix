with import <nixpkgs> {}; {
  sbt1311Env = stdenv.mkDerivation rec {
    name = "sbt-13-11-env";
    version = "1.0";
    src = ./.;
    buildInputs = [
        stdenv
        sbt
    ];
  };
}
