

class DataGen(Dataset):
    def __init__(self, image_paths, mask_paths, augment=False):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # load images
        image = Image.open(img_path).convert("F")   # float grayscale
        mask  = Image.open(mask_path).convert("L")  # label in 0/1

        # ----- simple augmentations -----
        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)

            if random.random() < 0.5:
                image = TF.vflip(image)
                mask  = TF.vflip(mask)

            # random 90° rotation
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                angle = 90 * k
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask  = TF.rotate(mask,  angle, interpolation=TF.InterpolationMode.NEAREST)

        # convert to tensor
        image = self.to_tensor(image)
        mask  = self.to_tensor(mask)

        # binarize mask
        mask = (mask > 0.5).float()

        # normalize (optional)
        # mean, std = image.mean(), image.std()
        # if std > 0:
        #     image = (image - mean) / std

        # 
        return {
            "image": image,
            "label": mask,
            "image_path": img_path,
        }
