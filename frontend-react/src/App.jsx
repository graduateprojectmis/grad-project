import { useState, useEffect } from 'react'
import Header from './components/Header'
import ChatContainer from './components/ChatContainer'
import InputArea from './components/InputArea'
import ApiKeyModal from './components/ApiKeyModal'
import WelcomeSection from './components/WelcomeSection'
import { checkApiHealth, checkApiKeyStatus } from './services/api'
import './index.css'

function App() {
  const [messages, setMessages] = useState([])
  const [isConnected, setIsConnected] = useState(false)
  const [apiKeyExists, setApiKeyExists] = useState(false)
  const [showApiKeyModal, setShowApiKeyModal] = useState(false)
  const [showWelcome, setShowWelcome] = useState(true)
  const [dbCount, setDbCount] = useState(0)
  const [currentCollection, setCurrentCollection] = useState('')

  useEffect(() => {
    // 初始檢查
    checkHealth()
    checkKeyStatus()
    
    // 定期檢查健康狀態
    const interval = setInterval(checkHealth, 30000)
    return () => clearInterval(interval)
  }, [])

  const checkHealth = async () => {
    try {
      const data = await checkApiHealth()
      setIsConnected(true)
      setDbCount(data.chroma_db_count || 0)
    } catch (error) {
      setIsConnected(false)
      console.error('API 連接失敗:', error)
    }
  }

  const checkKeyStatus = async () => {
    try {
      const data = await checkApiKeyStatus()
      setApiKeyExists(data.exists)
    } catch (error) {
      setApiKeyExists(false)
    }
  }

  const handleSendMessage = (question) => {
    setShowWelcome(false)
    setMessages(prev => [...prev, {
      id: Date.now(),
      content: question,
      isUser: true
    }])
  }

  const handleReceiveMessage = (answer, annotatedImages = []) => {
    setMessages(prev => [...prev, {
      id: Date.now(),
      content: answer,
      isUser: false,
      annotatedImages: annotatedImages
    }])
  }

  const handleReceiveLoadingMessage = (message) => {
    setMessages(prev => [...prev, message])
  }

  const handleRemoveMessage = (messageId) => {
    setMessages(prev => prev.filter(msg => msg.id !== messageId))
  }

  const handleGoHome = () => {
    setMessages([])
    setShowWelcome(true)
  }

  const handleCollectionChange = (collectionName, documentCount) => {
    setCurrentCollection(collectionName)
    setDbCount(documentCount)
    // 切換 collection 後清空訊息並返回首頁
    setMessages([])
    setShowWelcome(true)
    console.log(`已切換到：${collectionName}，文件數：${documentCount}`)
  }

  const handleExampleClick = async (question) => {
    // 檢查連接和 API Key
    if (!isConnected) {
      alert('❌ 無法連接到後端服務，請檢查後端是否正在運行！')
      return
    }

    if (!apiKeyExists) {
      alert('請先在伺服器設定 API Key！')
      setShowApiKeyModal(true)
      return
    }

    // 隱藏歡迎頁面並發送問題
    setShowWelcome(false)
    
    // 添加用戶問題和加載中的消息
    const loadingId = Date.now() + 1
    setMessages([
      {
        id: Date.now(),
        content: question,
        isUser: true
      },
      {
        id: loadingId,
        content: '🤔 正在思考中',
        isUser: false,
        isLoading: true
      }
    ])
    
    try {
      // 調用 API
      const { askQuestion } = await import('./services/api')
      const data = await askQuestion(question, 1)
      
      // 移除加載消息並添加答案
      setMessages(prev => [
        ...prev.filter(msg => msg.id !== loadingId),
        {
          id: Date.now() + 2,
          content: data.answer,
          isUser: false
        }
      ])
    } catch (error) {
      console.error('錯誤:', error)
      // 移除加載消息並顯示錯誤
      setMessages(prev => [
        ...prev.filter(msg => msg.id !== loadingId),
        {
          id: Date.now() + 2,
          content: `❌ 抱歉，發生錯誤：${error.message}<br>請稍後再試或檢查 API Key 是否正確。`,
          isUser: false
        }
      ])
    }
  }

  return (
    <div className="app-container">
      <Header
        isConnected={isConnected}
        dbCount={dbCount}
        onSettingsClick={() => setShowApiKeyModal(true)}
        onHomeClick={handleGoHome}
        onCollectionChange={handleCollectionChange}
      />
      
      <ChatContainer messages={messages}>
        {showWelcome && (
          <WelcomeSection onExampleClick={handleExampleClick} />
        )}
      </ChatContainer>
      
      <InputArea
        isConnected={isConnected}
        apiKeyExists={apiKeyExists}
        onSendMessage={handleSendMessage}
        onReceiveMessage={handleReceiveMessage}
        onReceiveLoadingMessage={handleReceiveLoadingMessage}
        onRemoveMessage={handleRemoveMessage}
        onOpenApiKeyModal={() => setShowApiKeyModal(true)}
      />
      
      {showApiKeyModal && (
        <ApiKeyModal
          onClose={() => setShowApiKeyModal(false)}
          onApiKeySaved={checkKeyStatus}
        />
      )}
    </div>
  )
}

export default App
