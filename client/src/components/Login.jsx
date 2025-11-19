import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import axios from 'axios';
import backgroundImage from '../assets/back2.jpg';

function Login() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');

        try {
            const response = await axios.post('http://localhost:5001/login', { email, password });
            localStorage.setItem('token', response.data.token); // Store the JWT token in localStorage
            navigate('/all-news');

        } catch (err) {
            setError(err.response.data.error || 'Login failed. Please try again.'); // Set the error message
        }
        console.log("success")
    };

    return (
        <div
            style={{
                backgroundImage: `url(${backgroundImage})`,
                backgroundSize: 'cover',
                backgroundPosition: 'center',
                height: '100vh',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                color: 'white',
            }}
        >
            <div className="bg-white p-8 rounded-lg shadow-2xl w-full max-w-sm ">
                <h2 className="text-2xl font-bold text-center mb-6" style={{ color: 'black' }}>Login</h2>
                {error && <p className="text-red-500 text-center font-bold">{error}</p>} {/* Error message */}
                <form onSubmit={handleSubmit}>
                    <div className="mb-4">
                        <label className="block text-gray-700 text-sm font-semibold mb-2" htmlFor="email" style={{ color: 'black', fontWeight: 'bold', fontSize: 16 }}>
                            Email
                        </label>
                        <input
                            type="email"
                            id="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                            className="w-full p-2 border border-gray-300 rounded focus:outline-none focus:border-blue-500"
                            placeholder="Enter your email"
                            style={{ color: 'black' }}
                        />
                    </div>
                    <div className="mb-6">
                        <label className="block text-gray-700 text-sm font-semibold mb-2" htmlFor="password" style={{ color: 'black', fontWeight: 'bold', fontSize: 16 }}>
                            Password
                        </label>
                        <input
                            type="password"
                            id="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                            className="w-full p-2 border border-gray-300 rounded focus:outline-none focus:border-blue-500"
                            placeholder="Enter your password"
                            style={{ color: 'black' }}
                        />
                    </div>
                    <button
                        type="submit"
                        className="w-full bg-blue-500 text-white p-2 rounded hover:bg-blue-600 transition duration-200 font-bold "
                        style={{ color: 'black', fontSize: 16 }}
                    >
                        Login
                    </button>
                </form>
                <p className="text-center text-sm mt-4 text-white font-bold" style={{ color: 'black', fontWeight: 'bold', fontSize: 14 }}>
                    If you are not already registered,{' '}
                    <Link to="/signup" className="text-blue-500 hover:underline font-bold">click here to sign up</Link>.
                </p>
            </div>
        </div>
    );
}

export default Login;
