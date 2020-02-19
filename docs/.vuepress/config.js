module.exports = {
    title: 'Triloon\'s Blog',
    description: 'Triloon: Say Hi',
    head: [['link', {rel: 'icon', href: '/img/eva.png'}],
           ['link', {rel: 'manifest', href: '/img/eva.png'}],
           ['link', {rel: 'apple-touch-icon', href: '/img/eva.png'}],
    ],
    serviceWorker: true,
    base: '/',
    markdown: {
        lineNumbers: true,
    },
    themeConfig: {
        logo: '/img/eva.png',
        nav: require('./nav.js'),
        sidebar: require('./sidebar.js'),
        sidebarDepth: 2,
        displayAllHeaders: true,
        lastUpdated: 'Last Updated',
        serviceWorker: {
            updatePopup: true,
        },
        /// read-only all webpages
        // repo: 'TriLoo/xblogs',
        // repoLabel: 'Contribute Code',
        // docsRepo: 'TriLoo/xblogs',
        // docsDir: 'docs',
        // docsBranch: 'master',
        // editLinks: false,
        // editLinkText: 'Edit this page',
    },
}
