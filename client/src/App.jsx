import { BrowserRouter, Route, Routes, useLocation } from "react-router-dom";
import AllNews from "./components/AllNews";
import TopHeadlines from "./components/TopHeadlines";
import ForYou from "./components/ForYou";
import Login from "./components/Login";
import SignUp from "./components/SignUp";
import Header from "./components/Header";
import React from "react";

function App() {
  return (
    <div className="w-full">
      <BrowserRouter>
        <MainContent />
      </BrowserRouter>
    </div>
  );
}

function MainContent() {
  const location = useLocation();

  // Check if the current path is either login or signup; if so, don't show the header
  const hideHeader = location.pathname === "/" || location.pathname === "/signup";

  return (
    <>
      {!hideHeader && <Header />}
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/signup" element={<SignUp />} />
        <Route path="/all-news" element={<AllNews />} />
        <Route path="/for-you" element={<ForYou />} />
        <Route path="/top-headlines/:category" element={<TopHeadlines />} />
      </Routes>
    </>
  );
}

export default App;
