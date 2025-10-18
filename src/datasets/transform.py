# datasets/transforms.py

import torchvision.transforms as T

class VideoTransforms:
    """
    Set of transformations for processing video frames.

    Provides standard transformations for training, validation, and testing
    models for violence detection in videos using 3D CNNs.

    Methods
    -------
    get_train_transforms(frame_size=(112, 112))
        Returns transformations to apply during training (includes data augmentation).

    get_val_transforms(frame_size=(112, 112))
        Returns transformations to apply during validation and testing (no augmentation).

    get_basic_transform(frame_size=(112, 112))
        Returns a basic transformation: ToTensor, Resize, and Normalize.
    """

    @staticmethod
    def get_train_transforms(frame_size=(112, 112)):
        """
        Returns transformations applied during training.

        Includes simple data augmentation to improve model generalization:
        brightness changes, horizontal flipping, and small rotations.

        Parameters
        ----------
        frame_size : tuple of int, optional
            Dimensions (height, width) to resize each frame to, by default (112, 112).

        Returns
        -------
        torchvision.transforms.Compose
            Composed transformation pipeline for training.
        """
        return T.Compose([
            T.ToPILImage(),
            T.Resize(frame_size),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomRotation(degrees=10),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    @staticmethod
    def get_val_transforms(frame_size=(112, 112)):
        """
        Returns transformations applied during validation or testing.

        No data augmentation is applied to ensure consistency across validation samples.

        Parameters
        ----------
        frame_size : tuple of int, optional
            Dimensions (height, width) to resize each frame to, by default (112, 112).

        Returns
        -------
        torchvision.transforms.Compose
            Composed transformation pipeline for validation or testing.
        """
        return T.Compose([
            T.ToPILImage(),
            T.Resize(frame_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    @staticmethod
    def get_basic_transform(frame_size=(112, 112)):
        """
        Returns a basic transformation pipeline without data augmentation.

        Useful for quick inference or very small datasets where augmentation
        may not be necessary.

        Parameters
        ----------
        frame_size : tuple of int, optional
            Dimensions (height, width) to resize each frame to, by default (112, 112).

        Returns
        -------
        torchvision.transforms.Compose
            Composed basic transformation pipeline.
        """
        return T.Compose([
            T.ToPILImage(),
            T.Resize(frame_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
