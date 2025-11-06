import { useState, useRef } from 'react'
import { Send, Paperclip, X } from 'lucide-react'
import { askQuestion, uploadImage } from '../services/api'
import './InputArea.css'

function InputArea({ 
  isConnected, 
  apiKeyExists, 
  onSendMessage, 
  onReceiveMessage,
  onOpenApiKeyModal 
}) {
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [selectedImage, setSelectedImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const fileInputRef = useRef(null)

  const handleSubmit = async (e) => {
    e?.preventDefault()
    
    const question = input.trim()
    if (!question && !selectedImage) return

    if (!isConnected) {
      alert('❌ 無法連接到後端服務，請檢查後端是否正在運行！')
      return
    }

    if (!apiKeyExists) {
      alert('請先在伺服器設定 API Key！')
      onOpenApiKeyModal()
      return
    }

    try {
      setIsLoading(true)

      // 如果有圖片，先上傳
      if (selectedImage) {
        onSendMessage(`📤 正在上傳圖片：${selectedImage.name}...`)
        
        const uploadData = await uploadImage(selectedImage)
        onReceiveMessage(
          `✅ ${uploadData.message}<br>` +
          `檔案名稱：${uploadData.filename}<br>` +
          `檔案大小：${formatFileSize(uploadData.file_size)}<br>` +
          `儲存路徑：${uploadData.file_path}`
        )
        
        clearImage()
        return
      }

      // 發送問題
      if (question) {
        onSendMessage(question)
        setInput('')
        
        const data = await askQuestion(question, 1)
        
        // 顯示答案
        onReceiveMessage(data.answer)
      }
    } catch (error) {
      console.error('錯誤:', error)
      onReceiveMessage(
        `❌ 抱歉，發生錯誤：${error.message}<br>` +
        '請稍後再試或檢查 API Key 是否正確。'
      )
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleImageSelect = (e) => {
    const file = e.target.files[0]
    if (!file) return

    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp']
    if (!allowedTypes.includes(file.type)) {
      alert('❌ 不支援的圖片格式！請選擇 JPG、PNG、GIF 或 WEBP 格式的圖片。')
      e.target.value = ''
      return
    }

    const maxSize = 5 * 1024 * 1024 // 5MB
    if (file.size > maxSize) {
      alert('❌ 圖片太大！請選擇小於 5MB 的圖片。')
      e.target.value = ''
      return
    }

    setSelectedImage(file)
    const reader = new FileReader()
    reader.onload = (e) => setImagePreview(e.target.result)
    reader.readAsDataURL(file)
  }

  const clearImage = () => {
    setSelectedImage(null)
    setImagePreview(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
  }

  const isDisabled = (!input.trim() && !selectedImage) || isLoading

  return (
    <div className="input-area">
      {imagePreview && (
        <div className="image-preview">
          <div className="preview-content">
            <img src={imagePreview} alt="預覽" />
            <div className="image-info">
              <span className="image-name">{selectedImage.name}</span>
              <span className="image-size">{formatFileSize(selectedImage.size)}</span>
            </div>
            <button 
              className="clear-button" 
              onClick={clearImage}
              title="移除圖片"
              aria-label="移除圖片"
            >
              <X size={16} />
            </button>
          </div>
        </div>
      )}
      
      <form className="input-container" onSubmit={handleSubmit}>
        <input
          type="file"
          ref={fileInputRef}
          accept="image/*"
          onChange={handleImageSelect}
          style={{ display: 'none' }}
        />
        
        <button
          type="button"
          className="upload-button"
          onClick={() => fileInputRef.current?.click()}
          title="上傳圖片"
          aria-label="上傳圖片"
        >
          <Paperclip size={20} />
        </button>
        
        <input
          type="text"
          className="question-input"
          placeholder="輸入您的問題..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={isLoading}
        />
        
        <button
          type="submit"
          className="send-button"
          disabled={isDisabled}
          aria-label="發送"
        >
          {isLoading ? (
            <span className="loading-text">思考中...</span>
          ) : (
            <>
              <Send size={18} />
              <span className="button-text">發送</span>
            </>
          )}
        </button>
      </form>
    </div>
  )
}

export default InputArea
