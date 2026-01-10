The iocamljs kernels, which are the main output of the project live here.

At the moment we generate 2 kernels;

`kernel.min.js` - minimum top level with (almost) nothing built in
`kernel.full.js` - top level with lwt, js\_of\_ocaml and camlp4 syntax extensions

The when used with iocamlserver it will know what kernel to run based on
command line arguments.

For use with an IPython installation rename one of these files to `kernel.js`
