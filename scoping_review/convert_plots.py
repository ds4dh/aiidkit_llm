import os
import sys
from playwright.sync_api import sync_playwright

# --- CONFIGURATION ---
# The directory containing your HTML files
TARGET_DIR = "scoping_review_results"

def convert_html_to_png(root_dir):
    # Verify directory exists
    if not os.path.exists(root_dir):
        print(f"Error: Directory not found: {root_dir}")
        return

    print("Starting browser engine...")
    
    with sync_playwright() as p:
        # Launch headless Chromium
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as e:
            print("Error launching browser. You might be missing system libraries.")
            print(f"Details: {e}")
            return

        count = 0
        
        # Walk through directory and subdirectories
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".html"):
                    html_path = os.path.join(subdir, file)
                    png_path = html_path.replace(".html", ".png")
                    
                    # Skip if PNG already exists to save time (optional)
                    # if os.path.exists(png_path): continue

                    print(f"Converting: {file} ...")
                    
                    try:
                        page = browser.new_page(viewport={'width': 1920, 'height': 1080})
                        # Load the local HTML file
                        page.goto(f"file://{os.path.abspath(html_path)}")
                        
                        # Wait for Plotly to finish animation/rendering
                        # 2000ms = 2 seconds. Increase if plots look empty.
                        page.wait_for_timeout(2000) 
                        
                        # Take screenshot
                        page.screenshot(path=png_path)
                        page.close()
                        count += 1
                    except Exception as e:
                        print(f"Failed to convert {file}: {e}")

        browser.close()
        print(f"\nSuccess! Converted {count} files.")

if __name__ == "__main__":
    convert_html_to_png(TARGET_DIR)