# Result Commands
These are the commands I used to generate my final results.

# Denoising

```bash
python3 denoising.py images/apple.jpg output/denoising-15 --noise 0.15 --no-verbose &
python3 denoising.py images/apple.jpg output/denoising-30 --noise 0.30 --no-verbose &
python3 denoising.py images/apple.jpg output/denoising-50 --noise 0.50 --no-verbose &
python3 denoising.py images/apple.jpg output/denoising-80 --noise 0.80 --no-verbose &
```

# Inpainting

```bash
python3 inpainting.py images/apple.jpg output/inpainting-15 --remove_percent 15 --no-verbose &
python3 inpainting.py images/apple.jpg output/inpainting-30 --remove_percent 30 --no-verbose &
python3 inpainting.py images/apple.jpg output/inpainting-50 --remove_percent 50 --no-verbose &
python3 inpainting.py images/apple.jpg output/inpainting-80 --remove_percent 80 --no-verbose &
```

# Segmentation

```bash
python3 segmentation.py images/apple.jpg output/segmentation-0/ --mask_coeff 999999 --no-verbose &
python3 segmentation.py images/apple2.png output/segmentation-1/ --no-verbose &
python3 segmentation.py images/shoe.png output/segmentation-2/ --mask_coeff 999999 --no-verbose &
python3 segmentation.py images/cats.jpg output/segmentation-3/ --mask_coeff 999999 --no-verbose &
```

# Transparency Separation

```bash
python3 transparency_separation.py images/curry-balloon output/transparent-0/ --noise 0.2 --no-verbose &
python3 transparency_separation.py images/strange-hamster output/transparent-1/ --noise 0.2 --no-verbose &
python3 transparency_separation.py images/strange-hamster-single output/transparent-2/ --no-verbose &
```

# Watermark Removal

```bash
python3 watermark.py images/watermark output/watermark-0 --no-verbose &
python3 watermark.py images/watermark2 output/watermark-1 --no-verbose &
python3 watermark.py images/watermark3 output/watermark-2 --no-verbose &
python3 watermark.py images/watermark_vecteezy/ output/watermark-3 --mask_coeff 0 --no-verbose &
```
