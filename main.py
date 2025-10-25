import pypandoc
import os

def md_to_pdf(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    pypandoc.convert_file(
        input_path,
        'pdf',
        outputfile=output_path,
        extra_args=[
            '--standalone',
            '--pdf-engine=xelatex',
            '--mathjax'
        ]
    )
    print(f"✅ PDF generated successfully: {output_path}")


if __name__ == "__main__":
    md_to_pdf("transformers.md", "transformers.pdf")
