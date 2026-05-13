# Contributing

These notes are field observations from one engineer running a specific
stack on specific hardware. I expect them to age: JetPack updates,
llama.cpp commits, NeMo versions, PyTorch fixes, all of these can change
what's true.

If you have:

- A reproduction of a finding on a different stack (different JetPack,
  different model, different Jetson SKU), open an issue or PR with your
  numbers. Cross-reproductions are the most valuable thing.
- A fix or workaround I missed, especially for the PyTorch NVML conflict
  in [docs/pytorch-nvml-conflict.md](docs/pytorch-nvml-conflict.md),
  please open a PR. Even pointing at an upstream issue or merged fix is
  enough.
- Updated numbers from a newer llama.cpp commit or JetPack version, a PR
  is welcome. Append your numbers to the relevant doc; don't replace
  mine, since the older data is useful for showing the trajectory.
- A correction: if something is plain wrong, file an issue with the
  evidence and I'll fix it. I tried to be honest but I wrote this while
  building a real product, so omissions are likely.

What I'm not going to do:

- Maintain this as a comprehensive Jetson edge-LLM resource. It's a
  focused log of what I learned while shipping. If it grows beyond that
  scope, I'll probably split it.
- Run benchmarks on hardware I don't own. I can't validate Orin NX or
  AGX Orin findings, but I'll gladly merge PRs from someone who can.
- Provide commercial support. If you need that, hire someone.

## License

By contributing you agree your contribution is licensed under the same
[Apache License 2.0](LICENSE) as the rest of the repo.
