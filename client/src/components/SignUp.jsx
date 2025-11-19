import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import axios from 'axios';
import backgroundImage from '../assets/back2.jpg';

function SignUp() {
    const [username, setUsername] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [showPreferences, setShowPreferences] = useState(false);
    const [selectedPreferences, setSelectedPreferences] = useState([]);
    const [token, setToken] = useState('');
    const navigate = useNavigate();

    const categories = [
        "business", "entertainment", "general", "health",
        "science", "sports", "technology", "politics"
    ];

    const handlePreferenceClick = (preference) => {
        setSelectedPreferences(prev => {
            if (prev.includes(preference)) {
                return prev.filter(p => p !== preference);
            } else {
                return [...prev, preference];
            }
        });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');

        try {
            // First sign up the user
            const signupResponse = await axios.post('http://localhost:5001/signup', {
                username,
                email,
                password
            });

            // Then log them in to get the token
            const loginResponse = await axios.post('http://localhost:5001/login', {
                email,
                password
            });

            setToken(loginResponse.data.token);
            localStorage.setItem('token', loginResponse.data.token);
            setShowPreferences(true);
        } catch (err) {
            setError(err.response?.data?.error || 'Signup failed. Please try again.');
        }
    };

    const handlePreferencesSubmit = async () => {
        try {
            await axios.post(
                'http://localhost:5001/setInitialPreferences',
                { preferences: selectedPreferences },
                {
                    headers: {
                        'Authorization': `Bearer ${token}`,
                        'Content-Type': 'application/json'
                    }
                }
            );
            navigate('/all-news');
        } catch (err) {
            setError(err.response?.data?.error || 'Failed to set preferences.');
        }
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
            {!showPreferences ? (

                <div className="bg-white p-8 rounded-lg shadow-md w-full max-w-sm">
                    <h2 className="text-2xl font-bold text-center mb-6 text-black " style={{ color: 'black' }}>Sign Up</h2>
                    {error && <p className="text-red-500 text-center" style={{ color: 'black', fontSize: 14, fontWeight: "bold" }}>{error}</p>} {/* Error message */}
                    <form onSubmit={handleSubmit}>
                        <div className="mb-4">
                            <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="username" style={{ color: 'black', fontSize: 16 }}>
                                Username
                            </label>
                            <input
                                type="text"
                                id="username"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                                required
                                className="w-full p-2 border border-gray-300 rounded focus:outline-none focus:border-blue-500"
                                placeholder="Enter your username"
                                style={{ color: 'black' }}
                            />
                        </div>
                        <div className="mb-4">
                            <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="email" style={{ color: 'black', fontSize: 16 }}>
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
                            <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="password" style={{ color: 'black', fontSize: 16 }}>
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
                            className="w-full bg-blue-500 text-white p-2 rounded hover:bg-blue-600 transition duration-200"
                            style={{ color: 'black', fontSize: 16, fontWeight: 'bold' }}
                        >
                            Sign Up
                        </button>
                    </form>
                    <p className="text-center text-sm mt-4" style={{ color: 'black', fontSize: 16, fontWeight: 'bold' }}>
                        Already have an account?{' '}
                        <Link to="/" className="text-blue-500 hover:underline">click here to login</Link>.
                    </p>
                </div>
            ) : (
                // Preferences selection popup
                <div className="bg-white p-8 rounded-lg shadow-md w-full max-w-md">
                    <h2 className="text-2xl font-bold text-center mb-6 text-black" style={{ color: 'black', fontSize: 20, fontWeight: 'bold' }}>
                        Select Your Preferences
                    </h2>
                    {error && <p className="text-red-500 text-center mb-4">{error}</p>}
                    <div className="grid grid-cols-2 gap-4 mb-6">
                        {categories.map((category) => (
                            <button
                                key={category}
                                onClick={() => handlePreferenceClick(category)}
                                className={`p-3 rounded-lg transition-colors duration-200 ${selectedPreferences.includes(category)
                                    ? 'bg-blue-600 text-white border-2 border-blue-700 shadow-lg transform scale-105'
                                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                                    }`}
                                style={{ color: 'black', fontSize: 16, fontWeight: 'bold' }}
                            >
                                {category.charAt(0).toUpperCase() + category.slice(1)}
                            </button>
                        ))}
                    </div>
                    <button
                        onClick={handlePreferencesSubmit}
                        disabled={selectedPreferences.length === 0}
                        className={`w-full p-3 rounded-lg text-white transition-colors duration-200 ${selectedPreferences.length === 0
                            ? 'bg-gray-400 cursor-not-allowed'
                            : 'bg-blue-500 hover:bg-blue-600'
                            }`}
                        style={{ color: 'black', fontSize: 16, fontWeight: 'bold' }}
                    >
                        Continue
                    </button>
                    <p className="text-gray-600 text-sm text-center mt-4" style={{ color: 'black', fontSize: 16, fontWeight: 'bold' }}>
                        Please select at least one preference to continue
                    </p>
                </div>
            )}
        </div>
    );
}

export default SignUp;
