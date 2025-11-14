import { useState, useRef } from 'react'
import { Send, Paperclip, X, Tag } from 'lucide-react'
import { askQuestion, uploadImage, annotateImage } from '../services/api'
import './InputArea.css'

function InputArea({ 
  isConnected, 
  apiKeyExists, 
  onSendMessage, 
  onReceiveMessage,
  onReceiveLoadingMessage,
  onRemoveMessage,
  onOpenApiKeyModal 
}) {
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [selectedImage, setSelectedImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [enableAnnotation, setEnableAnnotation] = useState(false)
  const [targetItem, setTargetItem] = useState('objects')
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

      // 如果有圖片，根據開關決定是否進行標註
      if (selectedImage) {
        if (enableAnnotation) {
          // 進行圖片標註
          onSendMessage(`� 正在標註圖片：${selectedImage.name}（偵測目標：${targetItem}）...`)
          
          const annotationData = await annotateImage(selectedImage, targetItem)
          
          let resultMessage = `✅ ${annotationData.message}<br>`
          resultMessage += `偵測到的物件：<br>`
          
          annotationData.objects.forEach((obj, idx) => {
            resultMessage += `${idx + 1}. ${obj.label} (座標: [${obj.box_2d.join(', ')}])<br>`
          })
          
          if (annotationData.annotated_images && annotationData.annotated_images.length > 0) {
            resultMessage += `<br>已儲存 ${annotationData.annotated_images.length} 個標註圖片`
          }
          
          onReceiveMessage(resultMessage)
        } else {
          // 單純上傳圖片
          onSendMessage(`�📤 正在上傳圖片：${selectedImage.name}...`)
          
          const uploadData = await uploadImage(selectedImage)
          onReceiveMessage(
            `✅ ${uploadData.message}<br>` +
            `檔案名稱：${uploadData.filename}<br>` +
            `檔案大小：${formatFileSize(uploadData.file_size)}<br>` +
            `儲存路徑：${uploadData.file_path}`
          )
        }
        
        clearImage()
        return
      }

      // 發送問題
      if (question) {
        onSendMessage(question)
        setInput('')
        
        // 添加加載中的消息
        const loadingId = Date.now() + 1
        onReceiveLoadingMessage({
          id: loadingId,
          content: '正在思考中',
          isUser: false,
          isLoading: true
        })
        
        const data = await askQuestion(question, 1)
        
        // 移除加載消息並顯示答案
        onRemoveMessage(loadingId)
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
          
          {/* 圖片標記開關 */}
          <div className="annotation-controls">
            <div className="annotation-toggle">
              <input
                type="checkbox"
                id="enable-annotation"
                checked={enableAnnotation}
                onChange={(e) => setEnableAnnotation(e.target.checked)}
              />
              <label htmlFor="enable-annotation">
                <Tag size={16} />
                啟用圖片標記
              </label>
            </div>
            
            {enableAnnotation && (
              <div className="target-item-input">
                <label htmlFor="target-item">偵測目標：</label>
                <input
                  type="text"
                  id="target-item"
                  value={targetItem}
                  onChange={(e) => setTargetItem(e.target.value)}
                  placeholder="例如：person, car, objects"
                />
              </div>
            )}
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
          className={`send-button ${isLoading ? 'loading' : ''}`}
          disabled={isDisabled}
          aria-label="發送"
        >
          {isLoading ? (
            <span className="loading-text">思考中</span>
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
