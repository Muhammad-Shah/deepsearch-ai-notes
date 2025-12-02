import os
import re
import sys
import shutil
import tempfile

def sanitize_math_escapes(text: str) -> str:
    # Remove backslashes that were used to escape characters that are valid in TeX math
    # but were mistakenly escaped (e.g. \_ \+ \=). Keep other TeX commands (e.g. \mu).
    return re.sub(r'\\([_=+])', r'\1', text)

def try_pypandoc(md_path: str, out_pdf: str) -> bool:
    try:
        import pypandoc
        pypandoc.convert_file(
            md_path,
            'pdf',
            outputfile=out_pdf,
            extra_args=['--standalone', '--pdf-engine=xelatex', '--mathjax']
        )
        print("✅ PDF generated successfully:", out_pdf)
        return True
    except Exception as e:
        last = str(e)
        print("pypandoc -> xelatex failed:", last.splitlines()[0])
        # detect common LaTeX 'Undefined control sequence' errors
        if 'Undefined control sequence' in last or 'xelatex not found' in last or 'Error producing PDF' in last:
            raise RuntimeError(last)
        return False

def fallback_html_or_pdf(md_path: str, out_pdf: str) -> bool:
    # Try to produce PDF via HTML route (wkhtmltopdf/pdfkit) or at least emit HTML
    try:
        import pypandoc
        html_out = os.path.splitext(out_pdf)[0] + ".html"
        pypandoc.convert_file(md_path, 'html', outputfile=html_out)
        # prefer wkhtmltopdf/pdfkit if available
        if shutil.which("wkhtmltopdf"):
            try:
                import pdfkit
                pdfkit.from_file(html_out, out_pdf, options={"enable-local-file-access": None})
                print("✅ PDF generated via HTML -> wkhtmltopdf:", out_pdf)
                return True
            except Exception as e:
                print("pdfkit/wkhtmltopdf step failed:", e)
        print("ℹ️  Could not create PDF with LaTeX; HTML written to:", html_out)
        return False
    except Exception as e:
        print("Fallback to HTML failed:", e)
        return False

def md_to_pdf(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    # First attempt: direct pypandoc -> xelatex
    try:
        if try_pypandoc(input_path, output_path):
            return
    except RuntimeError:
        # sanitize and retry
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        sanitized = sanitize_math_escapes(text)
        if sanitized == text:
            print("No simple math-escape fixes detected; not retrying sanitized file.")
        else:
            fd, tmp_md = tempfile.mkstemp(suffix=".md", text=True)
            os.close(fd)
            with open(tmp_md, "w", encoding="utf-8") as f:
                f.write(sanitized)
            try:
                print("Retrying conversion with sanitized markdown:", tmp_md)
                try:
                    if try_pypandoc(tmp_md, output_path):
                        os.remove(tmp_md)
                        return
                except RuntimeError:
                    pass
            finally:
                if os.path.exists(tmp_md):
                    os.remove(tmp_md)

    # Final fallback: HTML (and wkhtmltopdf/pdfkit if available)
    ok = fallback_html_or_pdf(input_path, output_path)
    if not ok:
        print("Conversion failed. Suggestions:")
        print(" - Fix math syntax in the markdown (use $...$ or \\[...\\] and TeX math: use _ not \\_ )")
        print(" - Install LaTeX (MiKTeX) and pandoc, then pip install pypandoc")
        print("     winget install --id=Pandoc.Pandoc -e")
        print("     winget install --id=MiKTeX.MiKTeX -e")
        print("     pip install pypandoc")
        print(" - Or install wkhtmltopdf and pip install pdfkit")
        print("     winget install --id=wkhtmltopdf.wkhtmltopdf -e")
        print("     pip install pdfkit")

if __name__ == "__main__":
    md_to_pdf("Machine Learning Optimizer Research Guide.md",
              "Machine Learning Optimizer Research Guide.pdf")