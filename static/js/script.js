// =========================
// ⚡ Spinner on Form Submit
// =========================
const form = document.getElementById('upload-form');
const spinner = document.getElementById('spinner');
let resultSection;

if (!form) {
    alert("Error: No form with id 'upload-form' found.");
} else {
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (spinner) spinner.style.display = 'block';
        const formData = new FormData(form);
        try {
            const response = await fetch("/process_ajax", {
                method: "POST",
                body: formData
            });
            if (!response.ok) {
                alert("Error submitting the form.");
                return;
            }
            const data = await response.json();

            // ✅ REMOVE ANY EXISTING RESULT
            if (resultSection) {
                resultSection.remove();
            }

            // ✅ CREATE RESULT SECTION
            resultSection = document.createElement("div");
            resultSection.innerHTML = `
                <h2>Result:</h2>
                <img src="${data.output_png}" alt="Output Image" class="image-preview"><br/>
                <div class="actions">
                    <a href="${data.output_png}" id="export-png" download>Export as .png</a>
                    <button id="export-txt">Export as .txt</button>
                    <button id="copy-text">Copy Text</button>
                </div>
                <div id="output-text" style="display:none;" class="preformatted-text">${data.text_result}</div>
            `;
            form.insertAdjacentElement('afterend', resultSection);

            attachExportAndCopy();
        } catch (error) {
            alert("Error submitting the form.");
            console.error(error);
        } finally {
            if (spinner) spinner.style.display = 'none';
        }
    });
}

// =========================
// 📄 Export and Copy Logic
// =========================
function attachExportAndCopy() {
    const exportTxtButton = document.getElementById('export-txt');
    const copyButton = document.getElementById('copy-text');

    exportTxtButton?.addEventListener('click', () => {
        const output = document.getElementById('output-text');
        saveText(output.textContent, 'output.txt');
    });
    copyButton?.addEventListener('click', async () => {
        const output = document.getElementById('output-text');
        try {
            await navigator.clipboard.writeText(output.textContent);
            alert("Text copied to clipboard!");
        } catch (error) {
            alert("Copy failed.");
        }
    });
}

function saveText(data, filename) {
    const blob = new Blob([data], { type: 'text/plain' });
    const a = document.createElement('a');
    a.download = filename;
    a.href = URL.createObjectURL(blob);
    a.click();
}

// =========================
// 🗑️ Clear File Button
// =========================
const clearButton = document.getElementById('clear_file');
const fileInput = document.querySelector('input[type="file"][name="file"]');
clearButton?.addEventListener('click', (e) => {
    e.preventDefault();
    if (fileInput) {
        fileInput.value = ""; // Clear the file
    }
});

// =========================
// 🎨 Toggle Custom Color Visibility Logic
// =========================
const colorMode = document.getElementById('color_mode');
const customColors = document.getElementById('custom_colors');
colorMode?.addEventListener('change', () => {
    customColors.style.display = colorMode.value === 'custom' ? 'block' : 'none';
});
if (colorMode.value === 'custom') {
    customColors.style.display = 'block';
}

// =========================
// 🌙 Theme Toggle Logic
// =========================
const themeButton = document.getElementById('theme-toggle');
themeButton?.addEventListener('click', () => {
    document.body.classList.toggle('light');
    localStorage.setItem('theme', document.body.classList.contains('light') ? 'light' : 'dark');
});
if (localStorage.getItem('theme') === 'light') {
    document.body.classList.add('light');
}