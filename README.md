<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Bezdarnost/claritycore/main/docs/assets/logo_full.png">
    <img alt="ClarityCore" src="https://raw.githubusercontent.com/Bezdarnost/claritycore/main/docs/assets/logo_full.png" width=100%>
  </picture>
</p>

<h3 align="center">
Easy and fast low-level vision for everyone
</h3>

---

Next-generation Open Source toolkit for low-level vision. Engineered for state-of-the-art performance in image and video pixel2pixel tasks, including Super-Resolution, Denoising, Deblurring, and more.

Quickstart (uv)
---------------

1. Ensure you have Python 3.10+ installed.
2. Install `uv` (one-time):

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Install the project in editable mode with uv:

```
uv pip install -e .
```

4. Run the sample entry point:

```
uv run python -m claritycore
```

Project Status
--------------

This project is in its early stage. APIs and structure will evolve rapidly.

Citation
--------

If you use ClarityCore in your research, please cite:

```
@software{ClarityCore_2025,
          author = {Urumbekov, Aman},
          license = {Apache-2.0},
          month = sep,
          title = {{ClarityCore}},
          url = {https://github.com/Bezdarnost/claritycore},
          year = {2025}
}
```

Urumbekov, A. (2025). ClarityCore: A toolkit for low-level vision. Version 0.0.1. GitHub. https://github.com/bezdarnost/claritycore

License
-------

Apache License 2.0. See `LICENSE`.
