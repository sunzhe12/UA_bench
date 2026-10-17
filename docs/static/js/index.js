// More Works Dropdown Functionality
function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (container && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');
    
    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

// Results carousel — self-contained replacement for the mismatched
// bulma-carousel bundle (the shipped JS was actually bulma-slider).
document.addEventListener('DOMContentLoaded', function() {
    const root = document.querySelector('.results-carousel');
    if (!root) return;

    const track = root.querySelector('.rc-track');
    const items = Array.from(track.children);
    const dotsWrap = root.querySelector('.rc-dots');
    const prevBtn = root.querySelector('.rc-prev');
    const nextBtn = root.querySelector('.rc-next');
    const reduceMotion = window.matchMedia &&
        window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    let index = 0;
    let timer = null;
    let visible = true;
    let hovered = false;

    const dots = items.map(function(_, i) {
        const dot = document.createElement('button');
        dot.type = 'button';
        dot.className = 'rc-dot' + (i === 0 ? ' is-active' : '');
        dot.setAttribute('aria-label', 'Show result figure ' + (i + 1));
        dot.addEventListener('click', function() { goTo(i); restart(); });
        dotsWrap.appendChild(dot);
        return dot;
    });

    function goTo(i) {
        index = (i + items.length) % items.length;
        track.style.transform = 'translateX(' + (-index * 100) + '%)';
        dots.forEach(function(d, j) { d.classList.toggle('is-active', j === index); });
    }

    function start() {
        if (reduceMotion || timer) return;
        timer = setInterval(function() { goTo(index + 1); }, 6000);
    }

    function stop() {
        if (timer) { clearInterval(timer); timer = null; }
    }

    function restart() { stop(); start(); }

    prevBtn.addEventListener('click', function() { goTo(index - 1); restart(); });
    nextBtn.addEventListener('click', function() { goTo(index + 1); restart(); });
    root.addEventListener('mouseenter', function() { hovered = true; stop(); });
    root.addEventListener('mouseleave', function() { hovered = false; if (visible) start(); });

    // Pause autoplay while the carousel is scrolled out of view
    if ('IntersectionObserver' in window) {
        new IntersectionObserver(function(entries) {
            entries.forEach(function(entry) {
                visible = entry.isIntersecting;
                if (visible && !hovered) { start(); } else { stop(); }
            });
        }, { threshold: 0.15 }).observe(root);
    }

    start();
});
