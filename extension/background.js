chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "checkDeepfake",
    title: "Check if Deepfake",
    contexts: ["image"]
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "checkDeepfake") {
    // Send message to content script to overlay a loading indicator, then we will process it
    chrome.tabs.sendMessage(tab.id, {
      action: "analyze_image",
      imageUrl: info.srcUrl
    });
  }
});
