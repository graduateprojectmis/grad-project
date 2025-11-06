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

  const handleReceiveMessage = (answer) => {
    setMessages(prev => [...prev, {
      id: Date.now(),
      content: answer,
      isUser: false
    }])
  }

  const handleGoHome = () => {
    setMessages([])
    setShowWelcome(true)
  }

  const handleExampleClick = (question) => {
    setShowWelcome(false)
  }

  return (
    <div className="app-container">
      <Header
        isConnected={isConnected}
        dbCount={dbCount}
        onSettingsClick={() => setShowApiKeyModal(true)}
        onHomeClick={handleGoHome}
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
