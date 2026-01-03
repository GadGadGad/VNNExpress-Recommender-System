import axios from 'axios';

const api = axios.create({
    baseURL: 'http://localhost:8000/api/v1',
    headers: {
        'Content-Type': 'application/json',
    },
});

export const endpoints = {
    config: {
        init: '/config/init',
        categories: '/config/categories',
    },
    recommend: {
        get: '/recommend/',
        search: '/recommend/search',
    },
    visuals: {
        embedding: '/visuals/embedding-space',
        wordcloud: '/visuals/wordcloud',
    }
};

export default api;
