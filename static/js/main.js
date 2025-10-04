/**
 * Main JavaScript file for Exoplanet Detection System
 * Handles common functionality across all pages
 */

// Global configuration
const API_BASE_URL = '/api';
const CHART_COLORS = {
    confirmed: '#28a745',
    candidate: '#ffc107', 
    false_positive: '#dc3545',
    primary: '#007bff',
    secondary: '#6c757d',
    info: '#17a2b8',
    warning: '#ffc107',
    danger: '#dc3545',
    success: '#28a745'
};

// Utility functions
const Utils = {
    /**
     * Format numbers with appropriate precision
     */
    formatNumber: function(num, precision = 2) {
        if (num === null || num === undefined || isNaN(num)) {
            return 'N/A';
        }
        
        if (Math.abs(num) >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (Math.abs(num) >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        } else {
            return parseFloat(num).toFixed(precision);
        }
    },

    /**
     * Format percentage values
     */
    formatPercentage: function(num, precision = 1) {
        if (num === null || num === undefined || isNaN(num)) {
            return 'N/A';
        }
        return (num * 100).toFixed(precision) + '%';
    },

    /**
     * Debounce function to limit API calls
     */
    debounce: function(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    /**
     * Show loading spinner
     */
    showLoading: function(elementId, message = 'Loading...') {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = `
                <div class="d-flex align-items-center justify-content-center">
                    <div class="spinner-border spinner-border-sm me-2" role="status"></div>
                    <span>${message}</span>
                </div>
            `;
        }
    },

    /**
     * Show error message
     */
    showError: function(elementId, message = 'An error occurred') {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    ${message}
                </div>
            `;
        }
    },

    /**
     * Show success message
     */
    showSuccess: function(elementId, message) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = `
                <div class="alert alert-success">
                    <i class="fas fa-check-circle me-2"></i>
                    ${message}
                </div>
            `;
        }
    },

    /**
     * Validate form data
     */
    validateForm: function(formData, requiredFields) {
        const errors = [];
        
        requiredFields.forEach(field => {
            if (!formData[field] || formData[field] === '') {
                errors.push(`${field.replace('_', ' ')} is required`);
            }
        });
        
        return errors;
    },

    /**
     * Convert form data to object
     */
    formDataToObject: function(formData) {
        const object = {};
        for (let [key, value] of formData.entries()) {
            if (value !== '') {
                // Try to convert to number if possible
                const numValue = parseFloat(value);
                object[key] = isNaN(numValue) ? value : numValue;
            }
        }
        return object;
    },

    /**
     * Download data as CSV
     */
    downloadCSV: function(data, filename) {
        const blob = new Blob([data], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    },

    /**
     * Download data as JSON
     */
    downloadJSON: function(data, filename) {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    }
};

// API helper functions
const API = {
    /**
     * Make API request with error handling
     */
    request: async function(endpoint, options = {}) {
        const url = `${API_BASE_URL}${endpoint}`;
        
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };
        
        const config = { ...defaultOptions, ...options };
        
        try {
            const response = await fetch(url, config);
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || `HTTP error! status: ${response.status}`);
            }
            
            return data;
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    },

    /**
     * Get system health status
     */
    getHealth: async function() {
        return await this.request('/health');
    },

    /**
     * Get model information
     */
    getModelInfo: async function() {
        return await this.request('/model/info');
    },

    /**
     * Get system statistics
     */
    getStats: async function() {
        return await this.request('/stats');
    },

    /**
     * Make single prediction
     */
    predict: async function(data) {
        return await this.request('/predict', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },

    /**
     * Make batch predictions
     */
    predictBatch: async function(samples) {
        return await this.request('/predict/batch', {
            method: 'POST',
            body: JSON.stringify({ samples })
        });
    },

    /**
     * Get feature template
     */
    getFeatureTemplate: async function() {
        return await this.request('/features/template');
    }
};

// Chart helper functions
const Charts = {
    /**
     * Create a pie chart
     */
    createPieChart: function(elementId, data, title = '') {
        const ctx = document.getElementById(elementId);
        if (!ctx) return null;
        
        return new Chart(ctx, {
            type: 'pie',
            data: {
                labels: data.labels,
                datasets: [{
                    data: data.values,
                    backgroundColor: data.colors || Object.values(CHART_COLORS).slice(0, data.labels.length),
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: !!title,
                        text: title
                    },
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    },

    /**
     * Create a bar chart
     */
    createBarChart: function(elementId, data, title = '') {
        const ctx = document.getElementById(elementId);
        if (!ctx) return null;
        
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.labels,
                datasets: [{
                    label: data.label || 'Values',
                    data: data.values,
                    backgroundColor: data.colors || CHART_COLORS.primary,
                    borderColor: data.borderColors || CHART_COLORS.primary,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: !!title,
                        text: title
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    },

    /**
     * Create a line chart
     */
    createLineChart: function(elementId, data, title = '') {
        const ctx = document.getElementById(elementId);
        if (!ctx) return null;
        
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: data.datasets.map((dataset, index) => ({
                    label: dataset.label,
                    data: dataset.data,
                    borderColor: dataset.color || Object.values(CHART_COLORS)[index],
                    backgroundColor: dataset.color || Object.values(CHART_COLORS)[index] + '20',
                    tension: 0.1,
                    fill: false
                }))
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: !!title,
                        text: title
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
};

// Notification system
const Notifications = {
    /**
     * Show toast notification
     */
    show: function(message, type = 'info', duration = 5000) {
        const toastContainer = this.getToastContainer();
        const toast = this.createToast(message, type);
        
        toastContainer.appendChild(toast);
        
        // Show toast
        setTimeout(() => {
            toast.classList.add('show');
        }, 100);
        
        // Auto hide
        setTimeout(() => {
            this.hide(toast);
        }, duration);
        
        return toast;
    },

    /**
     * Hide toast notification
     */
    hide: function(toast) {
        toast.classList.remove('show');
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    },

    /**
     * Get or create toast container
     */
    getToastContainer: function() {
        let container = document.getElementById('toast-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toast-container';
            container.className = 'position-fixed top-0 end-0 p-3';
            container.style.zIndex = '1050';
            document.body.appendChild(container);
        }
        return container;
    },

    /**
     * Create toast element
     */
    createToast: function(message, type) {
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type} border-0`;
        toast.setAttribute('role', 'alert');
        
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" onclick="Notifications.hide(this.closest('.toast'))"></button>
            </div>
        `;
        
        return toast;
    },

    /**
     * Show success notification
     */
    success: function(message, duration = 5000) {
        return this.show(message, 'success', duration);
    },

    /**
     * Show error notification
     */
    error: function(message, duration = 7000) {
        return this.show(message, 'danger', duration);
    },

    /**
     * Show warning notification
     */
    warning: function(message, duration = 6000) {
        return this.show(message, 'warning', duration);
    },

    /**
     * Show info notification
     */
    info: function(message, duration = 5000) {
        return this.show(message, 'info', duration);
    }
};

// File upload helper
const FileUpload = {
    /**
     * Handle file drag and drop
     */
    setupDragAndDrop: function(elementId, callback) {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        element.addEventListener('dragover', (e) => {
            e.preventDefault();
            element.classList.add('dragover');
        });
        
        element.addEventListener('dragleave', (e) => {
            e.preventDefault();
            element.classList.remove('dragover');
        });
        
        element.addEventListener('drop', (e) => {
            e.preventDefault();
            element.classList.remove('dragover');
            
            const files = Array.from(e.dataTransfer.files);
            callback(files);
        });
    },

    /**
     * Validate file type and size
     */
    validateFile: function(file, allowedTypes = ['.csv'], maxSize = 16 * 1024 * 1024) {
        const errors = [];
        
        // Check file type
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        if (!allowedTypes.includes(fileExtension)) {
            errors.push(`File type ${fileExtension} is not allowed. Allowed types: ${allowedTypes.join(', ')}`);
        }
        
        // Check file size
        if (file.size > maxSize) {
            errors.push(`File size (${Utils.formatNumber(file.size / 1024 / 1024)} MB) exceeds maximum allowed size (${Utils.formatNumber(maxSize / 1024 / 1024)} MB)`);
        }
        
        return errors;
    }
};

// Initialize common functionality when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Add smooth scrolling to anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.card');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    });
    
    cards.forEach(card => {
        observer.observe(card);
    });
    
    // Handle form validation
    const forms = document.querySelectorAll('.needs-validation');
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        });
    });
    
    // Auto-hide alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
    alerts.forEach(alert => {
        setTimeout(() => {
            if (alert.parentNode) {
                alert.style.opacity = '0';
                setTimeout(() => {
                    if (alert.parentNode) {
                        alert.parentNode.removeChild(alert);
                    }
                }, 300);
            }
        }, 5000);
    });
});

// Handle page visibility changes
document.addEventListener('visibilitychange', function() {
    if (document.visibilityState === 'visible') {
        // Page became visible, refresh data if needed
        if (typeof refreshPageData === 'function') {
            refreshPageData();
        }
    }
});

// Handle online/offline status
window.addEventListener('online', function() {
    Notifications.success('Connection restored');
});

window.addEventListener('offline', function() {
    Notifications.warning('Connection lost. Some features may not work properly.');
});

// Export utilities for global use
window.Utils = Utils;
window.API = API;
window.Charts = Charts;
window.Notifications = Notifications;
window.FileUpload = FileUpload;
