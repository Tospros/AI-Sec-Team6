Install PyTorch3D separately:
  pip install "git+https://github.com/facebookresearch/pytorch3d.git"
  # or: conda install pytorch3d -c pytorch3d

  Quick test (2D attack, no mesh needed):
  python demo_2d_attack.py --image turtle.jpg --target 764 --steps 200