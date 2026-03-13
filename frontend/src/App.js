/*
Infocreon Policy Assistant - React Frontend

This React application provides the chatbot interface where users
can ask questions about company HR policies.

Workflow:
1. User enters a question in the chatbot input field
2. The question is sent to the FastAPI backend using a POST request
3. The backend processes the query using the RAG engine
4. The generated answer and citation are returned to the frontend
5. The chatbot displays the answer along with the source document and page number

Features:
* Chat interface for user interaction
* API communication with backend server
* Displays AI responses and citations
* Upload PDF button for adding documents
*/
// Import React hooks
import React, { useState, useRef, useEffect } from "react";
// Import styling
import "./App.css";
// Import background and logo assets
import brochure from "./assets/brochure-bg-1.png";
import logo from "./assets/logo.png";

function App() {
// Store chat messages between user and chatbot
const [messages, setMessages] = useState([
{
  role: "bot",
  text: "Welcome to Infocreon Policy Assistant 🤖\n\nAn AI assistant designed to help you easily explore HR policies and company guidelines. Ask your question and I'll retrieve the most relevant information from our policy documents."
}
]);
// Store user input text
const [input, setInput] = useState("");
// Loading state while AI generates response
const [loading, setLoading] = useState(false);

const bottomRef = useRef(null);
const fileInputRef = useRef();
// Automatically scroll to the latest message when chat updates
useEffect(() => {
  bottomRef.current?.scrollIntoView({ behavior: "smooth" });
}, [messages, loading]);

// Function to send user question to backend API
const askQuestion = async () => {

  if (!input.trim()) return;

  const question = input;

  setMessages(prev => [...prev, { role: "user", text: question }]);
  setInput("");
  setLoading(true);

  try {

    const response = await fetch("http://127.0.0.1:8000/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });

    const data = await response.json();

    setMessages(prev => [
      ...prev,
      {
        role: "bot",
        text: `${data.answer}\n\nSource: ${data.source} (Page ${data.page})`
      }
    ]);

  } catch {

    setMessages(prev => [
      ...prev,
      { role: "bot", text: "Server error." }
    ]);

  }

  setLoading(false);
};

const uploadPDF = async (e) => {

  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  await fetch("http://127.0.0.1:8000/upload", {
    method: "POST",
    body: formData
  });

  alert("PDF uploaded successfully");
};

return (

<div
className="app"
style={{
  backgroundImage: `linear-gradient(rgba(0,35,90,0.7),rgba(0,35,90,0.7)),url(${brochure})`
}}
>

<div className="header">

<div className="header-left">
<img src={logo} alt="logo" />
<h3>Infocreon Policy Assistant</h3>
</div>

<button
className="upload-btn"
onClick={() => fileInputRef.current.click()}
>
Upload PDF
</button>

<input
type="file"
ref={fileInputRef}
style={{ display: "none" }}
onChange={uploadPDF}
/>

</div>

<div className="chat-area">

{messages.map((msg, index) => (

<div key={index} className={`message-row ${msg.role}`}>

<div className={`message-bubble ${msg.role}`}>

<div style={{ whiteSpace: "pre-line" }}>
{msg.text}
</div>

</div>

</div>

))}

{loading && (
<div className="message-row bot">
<div className="message-bubble bot">Thinking...</div>
</div>
)}

<div ref={bottomRef}></div>

</div>

<div className="input-bar">

<input
value={input}
placeholder="Ask your question..."
onChange={(e) => setInput(e.target.value)}
onKeyDown={(e) => {
  if (e.key === "Enter") askQuestion();
}}
/>

<button onClick={askQuestion}>
↑
</button>

</div>

</div>

);

}

export default App;