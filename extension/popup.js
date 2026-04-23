document.addEventListener('DOMContentLoaded', () => {
    const apiUrlInput = document.getElementById('apiUrl');
    const saveBtn = document.getElementById('saveBtn');
    const saveStatus = document.getElementById('saveStatus');

    chrome.storage.sync.get(['apiBaseUrl'], (result) => {
        apiUrlInput.value = result.apiBaseUrl || 'http://localhost:8000';
    });

    saveBtn.addEventListener('click', () => {
        const nextUrl = apiUrlInput.value.trim() || 'http://localhost:8000';
        chrome.storage.sync.set({ apiBaseUrl: nextUrl }, () => {
            saveStatus.textContent = 'Saved.';
        });
    });
});
