// 全域變數
let isApiConnected = false;
let selectedImage = null;

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    checkApiHealth();
    // 每 30 秒檢查一次連接狀態
    setInterval(checkApiHealth, 30000);
});

// 檢查 API 健康狀態
async function checkApiHealth() {
    const statusIndicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');
    
    try {
        const response = await fetch(
            `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.HEALTH}`
        );
        
        if (response.ok) {
            const data = await response.json();
            isApiConnected = true;
            statusIndicator.textContent = '🟢';
            statusText.textContent = `已連接 (${data.chroma_db_count} 筆資料)`;
            console.log('API 狀態:', data);
        } else {
            throw new Error('API 回應異常');
        }
    } catch (error) {
        isApiConnected = false;
        statusIndicator.textContent = '🔴';
        statusText.textContent = '連接失敗';
        console.error('無法連接到 API:', error);
        
        // 顯示錯誤訊息
        if (document.getElementById('chatContainer').children.length === 1) {
            addMessage(
                '⚠️ 無法連接到後端服務。請確認：<br>' +
                '1. 後端服務是否正在運行 (python api.py)<br>' +
                '2. API 地址是否正確<br>' +
                `3. 當前 API 地址：${API_CONFIG.BASE_URL}`,
                false,
                'error'
            );
        }
    }
}

// 添加訊息到聊天容器
function addMessage(content, isUser = false, type = 'normal') {
    const chatContainer = document.getElementById('chatContainer');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'bot'} ${type}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = content;
    
    messageDiv.appendChild(contentDiv);
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// 發送問題
async function askQuestion() {
    const input = document.getElementById('questionInput');
    const button = document.getElementById('askButton');
    const buttonText = document.getElementById('buttonText');
    const question = input.value.trim();
    
    // 檢查連接狀態
    if (!isApiConnected) {
        alert('❌ 無法連接到後端服務，請檢查後端是否正在運行！');
        return;
    }
    
    // 檢查問題是否為空
    if (!question) {
        alert('請輸入問題！');
        return;
    }
    
    // 顯示使用者問題
    addMessage(question, true);
    input.value = '';
    
    // 停用按鈕並顯示載入中
    button.disabled = true;
    buttonText.textContent = '思考中...';
    
    // 顯示載入動畫
    addMessage('<div class="loading-dots"><span></span><span></span><span></span></div>', false);
    
    try {
        const response = await fetch(
            `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.ASK}`,
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: question,
                    top_k: 1
                })
            }
        );
        
        // 移除載入動畫
        const chatContainer = document.getElementById('chatContainer');
        chatContainer.removeChild(chatContainer.lastChild);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        addMessage(data.answer);
        
    } catch (error) {
        // 移除載入動畫（如果還存在）
        const chatContainer = document.getElementById('chatContainer');
        if (chatContainer.lastChild.querySelector('.loading-dots')) {
            chatContainer.removeChild(chatContainer.lastChild);
        }
        
        console.error('錯誤:', error);
        addMessage(
            `❌ 抱歉，發生錯誤：${error.message}<br>` +
            '請稍後再試或聯繫管理員。',
            false,
            'error'
        );
    } finally {
        button.disabled = false;
        buttonText.textContent = '發送';
    }
}

// 使用範例問題
function askExample(question) {
    document.getElementById('questionInput').value = question;
    askQuestion();
}

// 處理 Enter 鍵
function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        askQuestion();
    }
}

// 處理圖片選擇
function handleImageSelect(event) {
    const file = event.target.files[0];
    
    if (!file) {
        return;
    }
    
    // 檢查檔案類型
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
        alert('❌ 不支援的圖片格式！請選擇 JPG、PNG、GIF 或 WEBP 格式的圖片。');
        event.target.value = '';
        return;
    }
    
    // 檢查檔案大小（限制 5MB）
    const maxSize = 5 * 1024 * 1024; // 5MB
    if (file.size > maxSize) {
        alert('❌ 圖片太大！請選擇小於 5MB 的圖片。');
        event.target.value = '';
        return;
    }
    
    selectedImage = file;
    
    // 顯示預覽
    const reader = new FileReader();
    reader.onload = function(e) {
        const previewDiv = document.getElementById('imagePreview');
        const previewImage = document.getElementById('previewImage');
        const imageName = document.getElementById('imageName');
        const imageSize = document.getElementById('imageSize');
        
        previewImage.src = e.target.result;
        imageName.textContent = file.name;
        imageSize.textContent = formatFileSize(file.size);
        previewDiv.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

// 清除選擇的圖片
function clearImage() {
    selectedImage = null;
    document.getElementById('imageInput').value = '';
    document.getElementById('imagePreview').style.display = 'none';
}

// 格式化檔案大小
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// 上傳圖片
async function uploadImage() {
    if (!selectedImage) {
        alert('請先選擇圖片！');
        return;
    }
    
    // 檢查連接狀態
    if (!isApiConnected) {
        alert('❌ 無法連接到後端服務，請檢查後端是否正在運行！');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', selectedImage);
    
    // 顯示上傳訊息
    addMessage(`📤 正在上傳圖片：${selectedImage.name}...`, true);
    
    // 顯示載入動畫
    addMessage('<div class="loading-dots"><span></span><span></span><span></span></div>', false);
    
    try {
        const response = await fetch(
            `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.UPLOAD}`,
            {
                method: 'POST',
                body: formData
            }
        );
        
        // 移除載入動畫
        const chatContainer = document.getElementById('chatContainer');
        chatContainer.removeChild(chatContainer.lastChild);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        addMessage(
            `✅ ${data.message}<br>` +
            `檔案名稱：${data.filename}<br>` +
            `檔案大小：${formatFileSize(data.file_size)}<br>` +
            `儲存路徑：${data.file_path}`
        );
        
        // 清除選擇的圖片
        clearImage();
        
    } catch (error) {
        // 移除載入動畫（如果還存在）
        const chatContainer = document.getElementById('chatContainer');
        if (chatContainer.lastChild.querySelector('.loading-dots')) {
            chatContainer.removeChild(chatContainer.lastChild);
        }
        
        console.error('上傳錯誤:', error);
        addMessage(
            `❌ 上傳失敗：${error.message}<br>` +
            '請檢查網絡連接或稍後再試。',
            false,
            'error'
        );
    }
}

// 修改發送按鈕功能，支援圖片上傳
const originalAskQuestion = askQuestion;
askQuestion = function() {
    // 如果有選擇圖片，則上傳圖片
    if (selectedImage) {
        uploadImage();
    } else {
        originalAskQuestion();
    }
};
