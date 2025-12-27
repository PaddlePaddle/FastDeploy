# 安装：pip install PyMuPDF pillow
import fitz  # PyMuPDF
import os

def extract_images_from_paper(pdf_path, output_dir="论文图片"):
    os.makedirs(output_dir, exist_ok=True)
    pdf = fitz.open(pdf_path)
    image_count = 0
    
    for page_num in range(len(pdf)):
        page = pdf[page_num]
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]  # 图片引用号
            base_image = pdf.extract_image(xref)
            
            if base_image:
                image_bytes = base_image["image"]
                ext = base_image["ext"]  # 原始格式：png、jpeg等
                img_filename = f"图{page_num+1}_{img_index+1}.{ext}"
                
                with open(f"{output_dir}/{img_filename}", "wb") as f:
                    f.write(image_bytes)
                image_count += 1
                print(f"已保存：{img_filename}")
    
    pdf.close()
    print(f"共提取 {image_count} 张图片")
    return image_count

# 使用
extract_images_from_paper("3728925.pdf")