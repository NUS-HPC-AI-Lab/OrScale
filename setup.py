from setuptools import find_packages, setup


INSTALL_REQUIRES = [
    "torch>=2.2.0",
    "numpy",
    "pyyaml",
    "tqdm",
]

EXTRAS_REQUIRE = {
    "analysis": ["matplotlib>=3.6.0", "scipy>=1.10.0"],
    "data": ["datasets", "huggingface-hub", "tiktoken"],
    "eval": ["lm-eval>=0.4.0", "tiktoken"],
    "vision": ["scipy>=1.10.0", "torchvision>=0.17.0"],
    "wandb": ["wandb"],
    "dev": ["pytest"],
}
EXTRAS_REQUIRE["all"] = sorted(
    {
        dep
        for extra, deps in EXTRAS_REQUIRE.items()
        if extra != "dev"
        for dep in deps
    }
)

setup(
    name="orscale",
    version="0.1.0",
    description="Orthogonalized updates with layer-wise scaling for language model training.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="NUS-HPC-AI-Lab",
    url="https://github.com/NUS-HPC-AI-Lab/OrScale",
    packages=find_packages(include=["orscale", "orscale.*"]),
    python_requires=">=3.9",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
