from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='bytetrack-gpu',
    version='1.0.0',
    description='GPU-accelerated multi-stream object tracking with ReID',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='sweetlhare',
    url='https://github.com/sweetlhare/BatchGpuByteTrack',
    license='MIT',
    packages=['bytetrack'],
    install_requires=[
        # torch>=1.13: torch.linalg.solve_triangular (1.11) and
        # torch.load(weights_only=True) (1.13)
        'torch>=1.13',
        'torchvision',
        'numpy',
        'scipy',
        'lap',
        'opencv-python',
    ],
    extras_require={
        'examples': ['ultralytics'],       # YOLO detector for examples/ and tools/
        'gdrive': ['gdown'],               # fallback source for OSNet weights
        'dev': ['pytest'],
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    project_urls={
        'Source': 'https://github.com/sweetlhare/BatchGpuByteTrack',
        'Bug Reports': 'https://github.com/sweetlhare/BatchGpuByteTrack/issues',
        'Original ByteTrack': 'https://github.com/FoundationVision/ByteTrack',
        'OSNet ReID': 'https://github.com/KaiyangZhou/deep-person-reid',
    },
)
