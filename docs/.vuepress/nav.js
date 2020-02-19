module.exports = [
    // each dict correspond to a folder
    {
        text: '计算机视觉', link: '/CVs/',
        items: [
            // each item correspond to a sidebar dict
            {text: 'Object Detection', link: '/CVs/ObjectDetection/'},
            {text: 'GAN', link: '/CVs/GAN/'},
            {text: 'SLAM', link: 'CVs/SLAM/'},
        ]
    },
    {
        text: '笔记', link: '/Notes/',
        items: [
            {text: '论文笔记', link: '/Notes/Papers/'},
            {text: '源码阅读', link: '/Notes/SourceCode/',
             items: [
                {text: 'MXNet', link: '/Notes/SourceCode/MXNet/'},
                {text: '经典GAN模型', link: '/Notes/SourceCode/GAN/'},
             ]
            },
            {text: 'Others', link: '/Notes/Others/'},
        ]
    },
    {
        text: 'Others', link: '/Others/',
    }
]
