import React, { useState, useRef, useEffect } from "react";
import styles from "./ChatTools.module.css";
import ReactMarkdown from "react-markdown";

function ChatTools() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [model, setModel] = useState("gpt-4o-mini");
  const containerRef = useRef(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [messages]); // runs when messages change

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    
    // Add user message to the chat
    setMessages((msgs) => [...msgs, { sender: "user", text: input }]);
    setLoading(true);
    
    // Create a new message for the bot response
    let botMsg = "";
    let botMsgIndex = null;
    
    try {
      // Make a POST request to the /chat/stream endpoint
      const response = await fetch('/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: input,
          model: model,
          max_tokens: 1000,
          temperature: 0.7
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      // Create a reader for the response stream
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      // Process the stream
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        // Decode the chunk
        const chunk = decoder.decode(value);
        
        // Split by lines and process each SSE event
        const lines = chunk.split('\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.substring(6).trim();
            
            // Check if it's the end of the stream
            if (data === '[DONE]') {
              setLoading(false);
              break;
            }
            
            try {
              // Parse the JSON data
              const jsonData = JSON.parse(data);
              
              // Handle regular content
              if (jsonData.content) {
                botMsg += jsonData.content;
                setMessages((msgs) => {
                  const last = msgs[msgs.length - 1];
                  if (last && last.sender === "bot" && botMsgIndex === msgs.length - 1) {
                    const updated = [...msgs];
                    updated[botMsgIndex] = { sender: "bot", text: botMsg };
                    return updated;
                  } else {
                    botMsgIndex = msgs.length;
                    return [...msgs, { sender: "bot", text: botMsg }];
                  }
                });
              }
              
              // Handle tool execution notifications
              if (jsonData.executing_tool) {
                setMessages((msgs) => [
                  ...msgs, 
                  { 
                    sender: "system", 
                    text: `Executing tool: ${jsonData.executing_tool}...`,
                    tool: jsonData.executing_tool
                  }
                ]);
              }
              
              // Handle tool detection
              if (jsonData.tool_calls_detected) {
                setMessages((msgs) => [
                  ...msgs, 
                  { 
                    sender: "system", 
                    text: "Tool calls detected. Processing..."
                  }
                ]);
              }
              
              // Handle tool results processed
              if (jsonData.tool_results_processed) {
                setMessages((msgs) => [
                  ...msgs, 
                  { 
                    sender: "system", 
                    text: "Tool results processed. Generating response..."
                  }
                ]);
                
                // Reset bot message for the final response
                botMsg = "";
                botMsgIndex = null;
              }
              
              // Handle errors
              if (jsonData.error) {
                setMessages((msgs) => [
                  ...msgs, 
                  { 
                    sender: "error", 
                    text: `Error: ${jsonData.error}`
                  }
                ]);
                setLoading(false);
              }
            } catch (e) {
              // Handle non-JSON data (should not happen with this endpoint)
              console.error("Failed to parse JSON:", e, data);
            }
          }
        }
      }
    } catch (error) {
      console.error("Error:", error);
      setMessages((msgs) => [
        ...msgs, 
        { 
          sender: "error", 
          text: `Error: ${error.message}`
        }
      ]);
      setLoading(false);
    }
    
    setInput("");
  };

  return (
    <div className={styles.chatContainer}>
      <div className={styles.header}>
        <h2>Azure OpenAI Chat</h2>
        <select 
          value={model} 
          onChange={(e) => setModel(e.target.value)}
          className={styles.modelSelect}
        >
          <option value="gpt-4o-mini">GPT-4o-mini</option>
          <option value="gpt-35-turbo">GPT-3.5 Turbo</option>
        </select>
      </div>
      <div className={styles.messages} ref={containerRef}>
        {messages.map((msg, i) => (
          <div key={i} className={styles.messageRow}>
            {msg.sender === "user" ? (
              <div className={styles.userMessage}><ReactMarkdown>{msg.text}</ReactMarkdown></div>
            ) : msg.sender === "system" ? (
              <div className={styles.systemMessage}>{msg.text}</div>
            ) : msg.sender === "error" ? (
              <div className={styles.errorMessage}>{msg.text}</div>
            ) : (
              <div className={styles.botMessage}><ReactMarkdown>{msg.text}</ReactMarkdown></div>
            )}
          </div>
        ))}
        {loading && (
          <div className={styles.loadingIndicator}>
            <div className={styles.loadingDot}></div>
            <div className={styles.loadingDot}></div>
            <div className={styles.loadingDot}></div>
          </div>
        )}
      </div>
      <form onSubmit={handleSend} className={styles.formRow}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          required
          className={styles.input}
          disabled={loading}
        />
        <button type="submit" disabled={loading} className={styles.button}>
          Send
        </button>
      </form>
    </div>
  );
}

export default ChatTools;
