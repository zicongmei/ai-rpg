const config = {
    type: Phaser.AUTO, // Use WebGL if available, otherwise Canvas
    width: 800,
    height: 600,
    physics: {
        default: 'arcade', // Arcade physics is good for platformers
        arcade: {
            gravity: { y: 800 }, // Adjust gravity as needed
            debug: false // Set to true to see physics bodies
        }
    },
    scene: {
        preload: preload,
        create: create,
        update: update
    },
    scale: {
        mode: Phaser.Scale.FIT, // Fit the game to the screen
        autoCenter: Phaser.Scale.CENTER_BOTH
    }
};

let game = new Phaser.Game(config);

function preload() {
    // Load assets here
    // Example: this.load.image('sky', 'assets/images/sky.png');
    // Example: this.load.spritesheet('mario', 'assets/images/mario-sprite.png', { frameWidth: 32, frameHeight: 48 });
    // Example: this.load.tilemapTiledJSON('level1', 'assets/tilemaps/level1.json');
    this.load.tilemapTiledJSON('level1', 'assets/tilemaps/level1.json');
}

let player;
let platforms;
let cursors;
let score = 0;
let scoreText;

function create() {
    // Create game elements here
    // Example: this.add.image(400, 300, 'sky');

    // Create platforms (static group for ground, etc.)
    platforms = this.physics.add.staticGroup();
    platforms.create(400, 568, 'ground').setScale(2).refreshBody();

    // Create player
    player = this.physics.add.sprite(100, 450, 'mario');
    player.setBounce(0.2);
    player.setCollideWorldBounds(true);

    // Add collision between player and platforms
    this.physics.add.collider(player, platforms);

    // Set up keyboard input
    cursors = this.input.keyboard.createCursorKeys();

    // Score text
    scoreText = this.add.text(16, 16, 'Score: 0', { fontSize: '32px', fill: '#000' });
}

function update() {
    // Game loop logic
    // Handle player movement
    if (cursors.left.isDown) {
        player.setVelocityX(-160);
        player.anims.play('left', true);
    } else if (cursors.right.isDown) {
        player.setVelocityX(160);
        player.anims.play('right', true);
    } else {
        player.setVelocityX(0);
        player.anims.play('turn');
    }

    if (cursors.up.isDown && player.body.touching.down) {
        player.setVelocityY(-330);
    }
}