# anime_dataset.py
"""
动画数据集对象体系（精简版）
层次: AnimeDataset → AnimeMovie → AnimeScene → AnimeFrame
只保留目录遍历和文件访问能力，操作接口预留为 NotImplementedError
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterator, Optional


class AnimeFrame:
    def __init__(self, frame_path: Path, scene: "AnimeScene", frame_id: int):
        self.frame_path = frame_path
        self.scene = scene
        self.frame_id = frame_id
        self.sample_key = str(
            frame_path.relative_to(scene.movie.dataset.root)
        ).replace("\\", "/")

    def get_bbox(self, **kwargs):
        raise NotImplementedError

    def get_skeleton(self, **kwargs):
        raise NotImplementedError

    def get_pose_data(self, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return f"AnimeFrame(id={self.frame_id:04d}, path={self.frame_path})"


class AnimeScene:
    def __init__(self, scene_dir: Path, movie: "AnimeMovie"):
        self.scene_dir = scene_dir
        self.movie = movie
        self.scene_name = scene_dir.name
        self.frames_dir = scene_dir / "frames"
        self.frame_data_dir = scene_dir / "frame_data"
        self.bbox_data_dir = scene_dir / "bbox_data"

    def to_frames(self, **kwargs) -> Path:
        raise NotImplementedError

    def iter_frames(self) -> Iterator[AnimeFrame]:
        if not self.frames_dir.exists():
            raise FileNotFoundError(f"frames 目录不存在，请先调用 to_frames(): {self.frames_dir}")
        for img_path in sorted(self.frames_dir.glob("*.png")):
            yield AnimeFrame(img_path, scene=self, frame_id=int(img_path.stem))

    def frame_count(self) -> int:
        if not self.frames_dir.exists():
            return 0
        return sum(1 for _ in self.frames_dir.glob("*.png"))

    def get_bbox(self, **kwargs):
        raise NotImplementedError

    def get_pose_data(self, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return f"AnimeScene(name={self.scene_name}, frames={self.frame_count()})"


class AnimeMovie:
    def __init__(self, movie_dir: Path, dataset: "AnimeDataset"):
        self.movie_dir = movie_dir
        self.dataset = dataset
        self.movie_name = movie_dir.name
        self.scenes_dir = self._find_scenes_dir()

    def _find_scenes_dir(self) -> Optional[Path]:
        for d in self.movie_dir.iterdir():
            if d.is_dir() and d.name.endswith("_scenes"):
                return d
        return None

    def iter_scenes(self) -> Iterator[AnimeScene]:
        if self.scenes_dir is None:
            return
        for d in sorted(self.scenes_dir.iterdir()):
            if d.is_dir():
                yield AnimeScene(d, movie=self)

    def iter_frames(self) -> Iterator[AnimeFrame]:
        for scene in self.iter_scenes():
            yield from scene.iter_frames()

    def scene_count(self) -> int:
        if self.scenes_dir is None:
            return 0
        return sum(1 for d in self.scenes_dir.iterdir() if d.is_dir())

    def to_frames(self, **kwargs):
        raise NotImplementedError

    def get_pose_data(self, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return f"AnimeMovie(name={self.movie_name}, scenes={self.scene_count()})"


class AnimeDataset:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"数据集根目录不存在: {self.root}")

    def iter_movies(self) -> Iterator[AnimeMovie]:
        for d in sorted(self.root.iterdir()):
            if d.is_dir():
                yield AnimeMovie(d, dataset=self)

    def iter_scenes(self, movie_name: Optional[str] = None) -> Iterator[AnimeScene]:
        for movie in self.iter_movies():
            if movie_name and movie.movie_name != movie_name:
                continue
            yield from movie.iter_scenes()

    def iter_frames(self, movie_name: Optional[str] = None) -> Iterator[AnimeFrame]:
        for movie in self.iter_movies():
            if movie_name and movie.movie_name != movie_name:
                continue
            yield from movie.iter_frames()

    def get_movie(self, movie_name: str) -> AnimeMovie:
        movie_dir = self.root / movie_name
        if not movie_dir.exists():
            raise FileNotFoundError(f"找不到动画目录: {movie_dir}")
        return AnimeMovie(movie_dir, dataset=self)

    def get_pose_data(self, **kwargs):
        raise NotImplementedError

    def movie_count(self) -> int:
        return sum(1 for d in self.root.iterdir() if d.is_dir())

    def __repr__(self):
        return f"AnimeDataset(root={self.root}, movies={self.movie_count()})"