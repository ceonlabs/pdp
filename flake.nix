{
 inputs = {
    mach-nix.url = "mach-nix/3.5.0";
  };

  outputs = {self, nixpkgs, mach-nix }@inp:
    let
      l = nixpkgs.lib // builtins;
      supportedSystems = [ "x86_64-linux" "aarch64-darwin" ];
      forAllSystems = f: l.genAttrs supportedSystems
        (system: f system (import nixpkgs {inherit system;}));
    in
    {
      # enter this python environment by executing `nix shell .`
      defaultPackage = forAllSystems (system: pkgs: mach-nix.lib."${system}".mkPython {
        requirements = ''
          numpy
          tqdm
          Cython
          flit-core
          pillow
        '';

        providers = {
          _default = "nixpkgs";
        };
        
        _.tinygrad.patches = [];
        #_.{package}.buildInputs = [...];             # replace buildInputs
        #_.{package}.buildInputs.add = [...];         # add buildInputs
        #_.{package}.buildInputs.mod =                # modify buildInputs
        #    oldInputs: filter (inp: ...) oldInputs;

        #_.{package}.patches = [...];                 # replace patches
        #_.{package}.patches.add = [...];             # add patches

      });
    };
}

