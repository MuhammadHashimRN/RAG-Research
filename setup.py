from setuptools import setup, find_packages

setup(
    name="agentic_rag",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "langchain",
        "faiss-cpu",
        "sentence-transformers",
        "torch",
        "numpy",
        "pyyaml",
    ],
)
