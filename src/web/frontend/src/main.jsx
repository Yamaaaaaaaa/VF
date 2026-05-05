import React from 'react';
import ReactDOM from 'react-dom/client';
import { createBrowserRouter, RouterProvider, Navigate } from 'react-router-dom';
import Layout from './layouts/Layout.jsx';
import SearchPage from './pages/SearchPage.jsx';
import DatabasePage from './pages/DatabasePage.jsx';
import './index.css';

const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      { index: true, element: <SearchPage /> },
      { path: 'database', element: <DatabasePage /> },
      // Fallback: redirect unknown paths to home
      { path: '*', element: <Navigate to="/" replace /> },
    ],
  },
]);

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);
